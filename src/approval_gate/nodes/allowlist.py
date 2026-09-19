"""Node 1: tool tiers and harm.

Each tool gets a harm level from what it does, not from which attacks use it:

  level 0  read-only                                            H 0.0
  level 1  internal, reversible write                           H 0.3
  level 2  affects other people, or commits money later         H 0.6
  level 3  irreversible in the account, or reaches outside      H 0.8
  level 4  irreversible transfer of money, data or access out   H 1.0

Tier: level 0 is read_only, 1 and 2 are write, 3 and 4 are
irreversible_external. Unknown tools get level 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Tier(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    IRREVERSIBLE_EXTERNAL = "irreversible_external"


HARM_BY_LEVEL = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.8, 4: 1.0}
UNKNOWN_LEVEL = 4

TOOL_LEVELS: dict[str, int] = {
    # ---- banking --------------------------------------------------------
    "get_iban": 0,
    "get_balance": 0,
    "get_most_recent_transactions": 0,
    "get_scheduled_transactions": 0,
    "get_user_info": 0,
    "read_file": 0,
    "update_user_info": 1,
    "schedule_transaction": 2,          # money leaves later, cancellable before then
    "update_scheduled_transaction": 2,  # same: redirects a future payment
    "send_money": 4,
    "update_password": 4,               # hands over account access
    # ---- workspace ------------------------------------------------------
    "get_unread_emails": 0,
    "get_sent_emails": 0,
    "get_received_emails": 0,
    "get_draft_emails": 0,
    "search_emails": 0,
    "search_contacts_by_name": 0,
    "search_contacts_by_email": 0,
    "get_current_day": 0,
    "search_calendar_events": 0,
    "get_day_calendar_events": 0,
    "search_files_by_filename": 0,
    "get_file_by_id": 0,
    "list_files": 0,
    "search_files": 0,
    "create_file": 1,
    "append_to_file": 1,
    "reschedule_calendar_event": 2,     # participants are notified
    "cancel_calendar_event": 2,
    "create_calendar_event": 2,         # invites participants
    "add_calendar_event_participants": 2,
    "delete_email": 3,
    "delete_file": 3,
    "send_email": 4,
    "share_file": 4,
    # ---- travel ---------------------------------------------------------
    "get_user_information": 0,
    "get_all_hotels_in_city": 0,
    "get_hotels_prices": 0,
    "get_rating_reviews_for_hotels": 0,
    "get_hotels_address": 0,
    "get_all_restaurants_in_city": 0,
    "get_cuisine_type_for_restaurants": 0,
    "get_restaurants_address": 0,
    "get_rating_reviews_for_restaurants": 0,
    "get_dietary_restrictions_for_all_restaurants": 0,
    "get_contact_information_for_restaurants": 0,
    "get_price_for_restaurants": 0,
    "check_restaurant_opening_hours": 0,
    "get_all_car_rental_companies_in_city": 0,
    "get_car_types_available": 0,
    "get_rating_reviews_for_car_rental": 0,
    "get_car_fuel_options": 0,
    "get_car_rental_address": 0,
    "get_car_price_per_day": 0,
    "get_flight_information": 0,
    "reserve_restaurant": 3,            # commits the user to an outside party
    "reserve_hotel": 3,
    "reserve_car_rental": 3,
    # ---- slack ----------------------------------------------------------
    "get_channels": 0,
    "read_channel_messages": 0,
    "read_inbox": 0,
    "get_users_in_channel": 0,
    "add_user_to_channel": 2,
    "get_webpage": 3,                   # read-shaped, but contacts an arbitrary
                                        # outside URL that can carry data out
    "send_channel_message": 3,
    "remove_user_from_slack": 3,
    "send_direct_message": 4,
    "invite_user_to_slack": 4,          # grants an outsider access
    "post_webpage": 4,
}


@dataclass(frozen=True)
class ToolInfo:
    level: int
    harm: float
    tier: Tier
    known: bool


def _tier(level: int) -> Tier:
    if level == 0:
        return Tier.READ_ONLY
    if level <= 2:
        return Tier.WRITE
    return Tier.IRREVERSIBLE_EXTERNAL


def classify(tool_name: str) -> ToolInfo:
    level = TOOL_LEVELS.get(tool_name, UNKNOWN_LEVEL)
    return ToolInfo(level=level, harm=HARM_BY_LEVEL[level], tier=_tier(level),
                    known=tool_name in TOOL_LEVELS)


def check_coverage(tool_names) -> None:
    """Fail if a suite has tools missing from the table."""
    missing = sorted(set(tool_names) - set(TOOL_LEVELS))
    if missing:
        raise SystemExit(f"nodes/allowlist.py has no entry for: {missing}")
