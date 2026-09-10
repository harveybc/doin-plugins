"""Eligibility VERIFICATION for DOIN plugins.

Per the work plan, `doin-plugins` verifies ids and digests. It
does not select and it does not promote: a plugin that finds a
subject ineligible refuses to run it, and a plugin that finds a
subject eligible has learned nothing it may act on beyond that
one permission.
"""
from doin_plugins.eligibility.adapter import (  # noqa: F401
    EligibilityUnavailable,
    load_gate,
)
from doin_plugins.eligibility.verify import (  # noqa: F401
    verify_identity,
)
