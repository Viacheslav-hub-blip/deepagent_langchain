import pandas as pd

record = {
    'epk_id': '2099007770421986000001',
    'event_dt': '20260309',
    'event_channel': 'CARDS',
    'transaction_amount_in_rub': 70986.27,
    'main_rule': '{"rule_name":"CARD_DENY_BLACK_LIST_FROM_DBO_2_group_potok","rule_id":"356b4d4c-abd6-48af-b86c-f675aba5a1b6","rule_category":"Behavior Anomaly"}',
    'policy_action': 'deny',
    'resolution_last': 'allow',
    'event_id': 'f9246b19-3bf5-4883-8076-d1d4356a6cf8'
}

target_hit_df = pd.DataFrame([{
    'event_id': record['event_id'],
    'epk_id': str(record['epk_id']),
    'event_dt': str(record['event_dt']),
    'event_channel': record['event_channel'],
    'amount': record['transaction_amount_in_rub'],
    'main_rule': record['main_rule'],
    'policy_action': record['policy_action'],
    'resolution': record['resolution_last']
}])

target_hit_df.to_csv('target_hit_df.csv', index=False)
print("DataFrame saved to target_hit_df.csv")
