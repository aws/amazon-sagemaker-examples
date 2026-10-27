from faker import Faker
import datetime
import itertools
import numpy as np
import pandas as pd
Faker.seed(0)
np.random.seed(0)

NUM_UNIQUE_CCS = 40*10**3
START_TRANS_DATE = datetime.datetime(2012, 1, 15)
END_TRANS_DATE = datetime.datetime(2012, 3, 15)

def gen_fraud_data(num_unique_ccs=NUM_UNIQUE_CCS, start_trans_date=START_TRANS_DATE, end_trans_date=END_TRANS_DATE):
    fake = Faker()
    cc_nums = [fake.credit_card_number() for _ in range(num_unique_ccs)]
    cc_types = [fake.credit_card_provider()for _ in range(num_unique_ccs)]
    num_trans_per_cc = np.ceil(np.random.exponential(scale=3, size=num_unique_ccs)).astype(np.int32)
    cc_ipv4 = [fake.ipv4() for _ in range(num_unique_ccs)]
    cc_phone_number = [fake.phone_number()for _ in range(num_unique_ccs)]
    cc_device_id = [fake.msisdn()for _ in range(num_unique_ccs)]

    data = {
        'TransactionID': [fake.uuid4() for _ in range(sum(num_trans_per_cc))],
        'TransactionDT': [fake.date_time_between_dates(datetime_start=start_trans_date, datetime_end=end_trans_date) 
                          for _ in range(sum(num_trans_per_cc))],
        'card_no': list(itertools.chain.from_iterable([[cc_num]*num_trans for cc_num, num_trans in zip(cc_nums, num_trans_per_cc)])),
        'card_type': list(itertools.chain.from_iterable([[card]*num_trans for card, num_trans in zip(cc_types, num_trans_per_cc)])),
        'email_domain': [fake.ascii_email().split("@")[1] for _ in range(sum(num_trans_per_cc))],
        'ProductCD': np.random.choice(['45', 'AB', 'L', 'Y', 'T'], size=sum(num_trans_per_cc)),
        'TransactionAmt': np.abs(np.ceil(np.random.exponential(scale=10, size=sum(num_trans_per_cc))*100)).astype(np.int32),
    }
    transactions = pd.DataFrame(data).sort_values(by=['TransactionDT'])
    
    # if you want to make the # of observations in the identity table less than that in the transactions table which may be more realistic in a practical scenario, change the size argument below.
    identity_transactions_idx = np.random.choice(transactions.shape[0], size=int(transactions.shape[0]*1.0), replace=False)
    id_data = {
        'IpAddress': list(itertools.chain.from_iterable([[ipv4]*num_trans for ipv4, num_trans in zip(cc_ipv4, num_trans_per_cc)])),
        'PhoneNo' : list(itertools.chain.from_iterable([[phone_num]*num_trans for phone_num, num_trans in zip(cc_phone_number, num_trans_per_cc)])),
        'DeviceID': list(itertools.chain.from_iterable([[device_id]*num_trans for device_id, num_trans in zip(cc_device_id, num_trans_per_cc)])),
    }
    identity = pd.DataFrame(id_data)
    identity["TransactionID"] = transactions.TransactionID
    assert identity.shape[0] == transactions.shape[0]
    
    identity = identity.loc[identity_transactions_idx]
    identity.reset_index(drop=True, inplace=True)
    identity = identity[["TransactionID", "IpAddress", "PhoneNo", "DeviceID"]]
    identity = pd.DataFrame(id_data)
    
    
    # join two tables for the convenience of generating label column 'isFraud'
    full_two_df = transactions[["TransactionID", "card_no", "card_type", "email_domain", "ProductCD", "TransactionAmt"]].merge(identity, on='TransactionID', how='left')

    is_fraud = []
    for idx, row in full_two_df.iterrows():
        card_no, card_type, email, product_type, transcation_amount, ip_address, phone_no, device_id = str(row["card_no"]), row["card_type"], row["email_domain"], row["ProductCD"], row["TransactionAmt"], str(row["IpAddress"]), str(row["PhoneNo"]), str(row["DeviceID"])
        
        if email in ["hotmail.com", "gmail.com", "yahoo.com"]:
            if product_type in ["45"]:
                is_fraud.append(int(np.random.uniform() < 0.9))
            else:
                if (device_id != "nan") and (device_id.endswith("16") or device_id.endswith("78") or device_id.endswith("23")):
                    is_fraud.append(int(np.random.uniform() < 0.1))
                else:
                    is_fraud.append(int(np.random.uniform() < 0.05))
        else:
            if transcation_amount > 3000:
                is_fraud.append(int(np.random.uniform() < 0.8))
            else:
                if card_type in ["Diners Club / Carte Blanche", "JCB 15 digit", "Maestro"]: # about 35,000 observations are in this categires
                    if (card_no.endswith("001") or card_no.endswith("002") or card_no.endswith("003") or card_no.endswith("004") or card_no.endswith("005") or card_no.endswith("007") or card_no.endswith("008") or card_no.endswith("009")) or ((phone_no != "nan") and (phone_no.endswith(".227") or phone_no.endswith(".104") or phone_no.endswith(".251") or phone_no.endswith(".181"))): 
                        is_fraud.append(int(np.random.uniform() < 0.3))
                    else:
                        if (ip_address != "nan") and (ip_address.endswith(".227") or ip_address.endswith(".104") or ip_address.endswith(".251") or ip_address.endswith(".181")):
                            is_fraud.append(int(np.random.uniform() < 0.2))
                        else:
                            is_fraud.append(int(np.random.uniform() < 0.1))
                else:
                    is_fraud.append(int(np.random.uniform() < 0.0001))
    print("fraud ratio", sum(is_fraud)/ len(is_fraud))
    
    transactions['isFraud'] = is_fraud
    return transactions, identity

if __name__ == '__main__':
    transaction, identity = gen_fraud_data()
    transaction.to_csv('raw_data/transaction.csv', index=False)
    identity.to_csv('raw_data/identity.csv', index=False)
