import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import time
from config import DROP_COLUMNS


class PreProcess:

    @staticmethod
    def fill_mort_acc(total_acc, mort_acc, total_acc_avg):
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc].round()
        else:
            return mort_acc

    def pre_process_pipeline(self):

        col_list = [
            "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti","earliest_cr_line",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
            "sub_grade_A2", "sub_grade_A3", "sub_grade_A4", "sub_grade_A5", "sub_grade_B1",
            "sub_grade_B2", "sub_grade_B3", "sub_grade_B4", "sub_grade_B5", "sub_grade_C1", "sub_grade_C2",
            "sub_grade_C3", "sub_grade_C4", "sub_grade_C5", "sub_grade_D1", "sub_grade_D2", "sub_grade_D3",
            "sub_grade_D4", "sub_grade_D5", "sub_grade_E1", "sub_grade_E2", "sub_grade_E3", "sub_grade_E4",
            "sub_grade_E5", "sub_grade_F1", "sub_grade_F2", "sub_grade_F3", "sub_grade_F4", "sub_grade_F5",
            "sub_grade_G1", "sub_grade_G2", "sub_grade_G3", "sub_grade_G4", "sub_grade_G5",
            "verification_status_Source Verified", "verification_status_Verified", "purpose_credit_card",
            "purpose_debt_consolidation", "purpose_educational", "purpose_home_improvement", "purpose_house",
            "purpose_major_purchase", "purpose_medical", "purpose_moving", "purpose_other",
            "purpose_renewable_energy", "purpose_small_business", "purpose_vacation", "purpose_wedding",
            "initial_list_status_w", "application_type_INDIVIDUAL", "application_type_JOINT",
            "home_ownership_MORTGAGE", "home_ownership_NONE", "home_ownership_OTHER", "home_ownership_OWN",
            "home_ownership_RENT", "zip_code_05113", "zip_code_11650", "zip_code_22690", "zip_code_29597",
            "zip_code_30723", "zip_code_48052", "zip_code_70466", "zip_code_86630", "zip_code_93700"]

        # Connection setup
        client = MongoClient("localhost", 27017)
        db = client["LendingClub"]
        cursor = db["staging"].find({})
        df = pd.DataFrame(list(cursor))
        df = df.drop("_id", axis=1)

        if len(df):
            # Append timestamp
            ts = time.time()
            df["added_timestamp"] = ts
            db.loan_data.insert_many(df.to_dict('records'))

            print("Processing new data")
            # Replace blank with nan
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Drop not required Features
            df.drop(DROP_COLUMNS, axis=1, inplace=True)

            # Filling mort_acc
            total_acc_avg = df.groupby(by='total_acc').mean().mort_acc
            df.mort_acc = df.apply(lambda x: PreProcess.fill_mort_acc(x['total_acc'], x['mort_acc'], total_acc_avg),
                                   axis=1)

            # drop rest of nan
            df.dropna(inplace=True)

            # Encoding term
            term_values = {' 36 months': 36, ' 60 months': 60}
            df['term'] = df.term.map(term_values)

            df['zip_code'] = df.address.apply(lambda x: x[-5:])
            df.drop("address", axis=1, inplace=True)

            # Filtering earliest_cr_line
            df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
            df['earliest_cr_line'] = df.earliest_cr_line.dt.year

            dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status',
                       'application_type', 'home_ownership', 'zip_code']

            buffer = {}
            for index, row in df[dummies].iterrows():
                for i in dummies:
                    if "{}_{}".format(i, row[i]) in col_list:
                        if "{}_{}".format(i,row[i]) not in buffer:
                            buffer["{}_{}".format(i, row[i])] = []
                            buffer["{}_{}".format(i,row[i])].append(index)
                        else:
                            buffer["{}_{}".format(i, row[i])].append(index)

            for k, v in buffer.items():
                df[k] = 0
                for j in v:
                    df.loc[j, k] = 1

            for i in col_list:
                if i not in df:
                    df[i] = 0

            df = df[col_list]

            # Insert and delete
            db.pre_processed_data.insert_many(df.to_dict('records'))
            db.staging.delete_many({})
            return df
        else:
            print("No New Data")

    def pre_processing(self):

        # Connection setup
        client = MongoClient("localhost", 27017)
        db = client["LendingClub"]
        cursor = db["staging"].find({})
        df = pd.DataFrame(list(cursor))

        if len(df):

            # Append timestamp
            ts = time.time()
            df["added_timestamp"] = ts
            db.loan_data.insert_many(df.to_dict('records'))

            print("Processing new data")
            # Replace blank with nan
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Drop not required Features
            df.drop(DROP_COLUMNS, axis=1, inplace=True)

            # Filling mort_acc
            total_acc_avg = df.groupby(by='total_acc').mean().mort_acc
            df.mort_acc = df.apply(lambda x: PreProcess.fill_mort_acc(x['total_acc'], x['mort_acc'], total_acc_avg),
                                   axis=1)

            # drop rest of nan
            df.dropna(inplace=True)

            # Encoding term
            term_values = {' 36 months': 36, ' 60 months': 60}
            df['term'] = df.term.map(term_values)

            # Generating dummies values
            dummies = ['sub_grade', 'verification_status', 'purpose', 'initial_list_status',
                       'application_type', 'home_ownership']
            df = pd.get_dummies(df, columns=dummies, drop_first=True)

            # Filtering Zip code
            df['zip_code'] = df.address.apply(lambda x: x[-5:])

            # Address similar to zip
            df.drop("address", axis=1, inplace=True)

            # Dummies for zip
            df = pd.get_dummies(df, columns=['zip_code'], drop_first=True)

            # Filtering earliest_cr_line
            df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
            df['earliest_cr_line'] = df.earliest_cr_line.dt.year

            # Insert and delete
            db.pre_processed_data.insert_many(df.to_dict('records'))
            db.staging.delete_many({})
            return df
        else:
            print("No New Data")

        """cursor = db["pre_processed_data"].find({})
        df = pd.DataFrame(list(cursor))
        return df"""
