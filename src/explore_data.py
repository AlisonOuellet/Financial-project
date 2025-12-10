from ydata_profiling import ProfileReport
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Dataset, DataDefinition, Report
import os
from evidently import Report
import os

def summarize_data_to_html(data, title, save_path):
    profile = ProfileReport(data, title=title, explorative=True)
    profile.to_file(save_path)
    print(f"Report saved : {save_path}")

import matplotlib.pyplot as plt
import pandas as pd

def get_drift(data_train, data_oos, data_oot, data_oou, output_path):

    sets = {
        "Train": data_train,
        "OOS": data_oos,
        "OOT": data_oot,
        "OOU": data_oou
        }

    dist = {
        name: df["DFlag"].mean() * 100
        for name, df in sets.items()
    }

    drift_majority = {
        "Train–OOS": 0,
        "Train–OOT": 5,
        "Train–OOU": 2
    }

    drift_minority = {
        "Train–OOS": 0,
        "Train–OOT": 2,
        "Train–OOU": 2
    }

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]

    ax1.bar(dist.keys(), dist.values())
    ax1.set_title("Proportion de défaut (DFlag=1) par jeu de données")
    ax1.set_ylabel("% classe minoritaire")
    ax1.set_ylim(0, max(dist.values()) * 1.25)

    for i, v in enumerate(dist.values()):
        ax1.text(i, v + 0.1, f"{v:.2f}%", ha='center')

    ax2 = axes[1]

    x = list(drift_majority.keys())
    maj = list(drift_majority.values())
    mino = list(drift_minority.values())

    ax2.bar([p - 0.15 for p in range(len(x))], maj, width=0.3, label="Majorité")
    ax2.bar([p + 0.15 for p in range(len(x))], mino, width=0.3, label="Minorité")

    ax2.set_xticks(range(len(x)))
    ax2.set_xticklabels(x)
    ax2.set_title("Nombre de variables présentant un drift")
    ax2.set_ylabel("Nombre de variables")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return dist, drift_majority, drift_minority


def run_drift_reports(
    data_train,
    data_oos,
    data_oot,
    data_oou,
    output_path="outputs/drift"):

    CATEGORICAL_GROUPS = {
        "Occupancy_status": [
            "is_Occupancy_status_prim",
            "is_Occupancy_status_inve",
            "is_Occupancy_status_seco"
        ],
        "First_time_homeowner": [
            "is_First_time_homeowner",
            "is_First_time_homeowner_No"
        ],
        "Property_type": [
            "is_Property_type_sing",
            "is_Property_type_pud",
            "is_Property_type_cond",
            "is_Property_type_coop",
            "is_Property_type_manu"
        ],
        "Origination_channel": [
            "is_Origination_channel_reta",
            "is_Origination_channel_corr",
            "is_Origination_channel_brok",
            "is_Origination_channel_tpo"
        ],
        "Loan_purpose": [
            "is_Loan_purpose_purc",
            "is_Loan_purpose_cash",
            "is_Loan_purpose_noca"
        ]
    }

    NUMERIC_COLUMNS = [
        "Credit_Score", "Mortgage_Insurance", "Number_of_units",
        "CLoan_to_value", "Debt_to_income", "OLoan_to_value"
    ]

    CATEGORICAL_COLUMNS = [
        'DFlag',
        'Occupancy_status', 'First_time_homeowner', 'Property_type',
        'Origination_channel', 'Loan_purpose', 'Single_borrower'
    ]

    TARGET_COLUMN = "DFlag"

    def one_hot_to_category(df):
        df = df.copy()
        for cat_name, cols in CATEGORICAL_GROUPS.items():
            existing = [c for c in cols if c in df.columns]
            if not existing:
                continue

            def get_cat(row):
                for c in existing:
                    if row[c] == 1.0:
                        return c.replace(f"is_{cat_name}_", "")
                return "unknown"

            df[cat_name] = df.apply(get_cat, axis=1)
        return df

    def drop_one_hot_columns(df):
        to_drop = [c for cols in CATEGORICAL_GROUPS.values() for c in cols if c in df.columns]
        return df.drop(columns=to_drop, errors="ignore")

    def clean(df):
        return drop_one_hot_columns(one_hot_to_category(df))

    train_clean = clean(data_train)
    oos_clean   = clean(data_oos)
    oot_clean   = clean(data_oot)
    oou_clean   = clean(data_oou)

    train_maj = clean(data_train[data_train[TARGET_COLUMN] == 0])
    train_min = clean(data_train[data_train[TARGET_COLUMN] == 1])

    oos_maj = clean(data_oos[data_oos[TARGET_COLUMN] == 0])
    oos_min = clean(data_oos[data_oos[TARGET_COLUMN] == 1])

    oot_maj = clean(data_oot[data_oot[TARGET_COLUMN] == 0])
    oot_min = clean(data_oot[data_oot[TARGET_COLUMN] == 1])

    oou_maj = clean(data_oou[data_oou[TARGET_COLUMN] == 0])
    oou_min = clean(data_oou[data_oou[TARGET_COLUMN] == 1])

    definition = DataDefinition(
        numerical_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS
    )

    os.makedirs(output_path, exist_ok=True)

    def run_report(name, ref, cur):
        report = Report(metrics=[DataDriftPreset()])
        report = report.run(
            reference_data=Dataset.from_pandas(ref, data_definition=definition),
            current_data=Dataset.from_pandas(cur, data_definition=definition)
        )
        report.save_html(f"{output_path}/{name}.html")

    run_report("train_vs_oos", train_clean, oos_clean)
    run_report("train_vs_oot", train_clean, oot_clean)
    run_report("train_vs_oou", train_clean, oou_clean)

    run_report("majority_train_vs_oos", train_maj, oos_maj)
    run_report("majority_train_vs_oot", train_maj, oot_maj)
    run_report("majority_train_vs_oou", train_maj, oou_maj)

    run_report("minority_train_vs_oos", train_min, oos_min)
    run_report("minority_train_vs_oot", train_min, oot_min)
    run_report("minority_train_vs_oou", train_min, oou_min)
