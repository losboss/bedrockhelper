"""
AWS credentials and client construction for BedrockHelper.

Kept apart from the Bedrock calls because it decides whose identity every
request runs under: changes here deserve a credentials-minded review.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import boto3
from botocore.config import Config
from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials
from botocore.exceptions import BotoCoreError, ClientError
from botocore.session import Session as BotocoreSession

log = logging.getLogger(__name__)

_EXPIRED_TOKEN_CODES = frozenset(
	{
		'ExpiredToken',
		'ExpiredTokenException',
		'InvalidClientTokenId',
		'UnrecognizedClientException',
		'RequestExpired',
	}
)


def is_expired_token(err: BaseException) -> bool:
	if isinstance(err, ClientError):
		code = (err.response.get('Error') or {}).get('Code')
		return code in _EXPIRED_TOKEN_CODES
	return False


def is_session_of_role(caller_arn: str, role_arn: str) -> bool:
	"""
	Whether `caller_arn` (from sts:GetCallerIdentity) is a session of the IAM role `role_arn`.

	The two ARNs never share a substring, so they are compared field by field:

	  caller: arn:aws:sts::123456789012:assumed-role/RoleName/session-name
	  role:   arn:aws:iam::123456789012:role/optional/path/RoleName

	The assumed-role ARN drops the role's path, so only the last path segment
	of the role name is compared.
	"""
	caller = caller_arn.split(':', 5)
	role = role_arn.split(':', 5)
	if len(caller) != 6 or len(role) != 6:
		return False
	_, caller_partition, caller_service, _, caller_account, caller_resource = caller
	_, role_partition, role_service, _, role_account, role_resource = role
	if caller_service != 'sts' or role_service != 'iam':
		return False
	if caller_partition != role_partition or caller_account != role_account:
		return False

	caller_parts = caller_resource.split('/')
	role_parts = role_resource.split('/')
	if len(caller_parts) != 3 or caller_parts[0] != 'assumed-role':
		return False
	if len(role_parts) < 2 or role_parts[0] != 'role':
		return False
	return caller_parts[1] == role_parts[-1]


class BotoSessionManager:
	"""
	Thread-safe manager that provides boto3 clients using:
	  - Default AWS credential resolution (auto-refresh when supported), OR
	  - Refreshable AssumeRole credentials (botocore DeferredRefreshableCredentials)

	Also supports invalidation to rebuild session/clients on ExpiredToken errors.
	"""

	def __init__(
		self,
		*,
		region_name: str,
		botocore_config: Config,
		role_arn: str | None = None,
		role_session_name: str = 'bedrockhelper',
		external_id: str | None = None,
		sts_region_name: str | None = None,
		assume_role_duration_seconds: int = 3600,
	) -> None:
		self._region = region_name
		self._cfg = botocore_config
		self._role_arn = role_arn
		self._role_session_name = role_session_name
		self._external_id = external_id
		self._sts_region = sts_region_name or region_name
		self._duration = int(assume_role_duration_seconds) if assume_role_duration_seconds else 3600

		self._lock = threading.RLock()
		self._boto3_session: boto3.Session | None = None
		self._clients: dict[str, Any] = {}

	def invalidate(self) -> None:
		with self._lock:
			self._boto3_session = None
			self._clients.clear()

	def _is_already_assumed_role(self, target_role_arn: str) -> bool:
		"""
		Whether the default credentials already belong to the target role.

		This matters because the role ARN defaults to AWS_ROLE_ARN, which EKS
		(IRSA) also sets: there the process already runs as the role, and
		assuming it again needs a self-trust policy most roles lack.
		"""
		sts = boto3.Session(region_name=self._sts_region).client('sts', config=self._cfg)
		try:
			current_arn = sts.get_caller_identity().get('Arn', '')
		except (BotoCoreError, ClientError) as e:
			log.warning('Could not read the current AWS identity (%s); assuming role %s.', e, target_role_arn)
			return False
		return is_session_of_role(current_arn, target_role_arn)

	def _build_boto3_session(self) -> boto3.Session:
		# No explicit role -> rely on normal resolution (IMDS/IRSA/ECS/SSO/etc). Refresh handled by botocore.
		if not self._role_arn:
			return boto3.Session(region_name=self._region)

		# Build botocore session with refreshable assume-role credentials.
		bc = BotocoreSession()
		bc.set_config_variable('region', self._region)

		# Ensure source credentials are loaded (refreshable if the provider supports it).
		source_creds = bc.get_credentials()
		if source_creds is None:
			raise RuntimeError('Unable to resolve AWS source credentials for AssumeRole')

		if self._is_already_assumed_role(self._role_arn):
			log.info('Already running as %s; not assuming it again.', self._role_arn)
			return boto3.Session(region_name=self._region)

		# STS client created from a standard boto3 session using source creds.
		sts = boto3.Session(region_name=self._sts_region).client('sts', config=self._cfg)

		extra_args: dict[str, Any] = {'RoleSessionName': self._role_session_name}
		if self._external_id:
			extra_args['ExternalId'] = self._external_id
		if self._duration:
			extra_args['DurationSeconds'] = self._duration

		fetcher = AssumeRoleCredentialFetcher(
			client_creator=lambda service_name, region_name=self._sts_region, **kwargs: sts,
			source_credentials=source_creds,
			role_arn=self._role_arn,
			extra_args=extra_args,
		)

		refreshable = DeferredRefreshableCredentials(
			method='assume-role',
			refresh_using=fetcher.fetch_credentials,
		)

		# Attach refreshable creds to botocore session. botocore has no public setter for a
		# credentials object; set_credentials() would replace it with a static snapshot that
		# never refreshes, so the session would fail once the first AssumeRole creds expire.
		bc._credentials = refreshable  # type: ignore[attr-defined]  # noqa: SLF001

		return boto3.Session(botocore_session=bc, region_name=self._region)

	def _ensure_session(self) -> boto3.Session:
		with self._lock:
			if self._boto3_session is None:
				self._boto3_session = self._build_boto3_session()
				self._clients.clear()
			return self._boto3_session

	def client(self, service_name: str) -> Any:
		with self._lock:
			sess = self._ensure_session()
			if service_name not in self._clients:
				self._clients[service_name] = sess.client(  # type: ignore[call-overload]
					service_name,
					region_name=self._region,
					config=self._cfg,
				)
			return self._clients[service_name]
