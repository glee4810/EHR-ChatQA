DROP TABLE IF EXISTS icupatient;
CREATE TABLE icupatient
(
    patient_id TEXT NOT NULL,
    hosp_id INT NOT NULL,
    unit_id INT NOT NULL PRIMARY KEY,
    gender TEXT NOT NULL,
    age INT NOT NULL,
    ethnicity TEXT,
    hospital_id INT NOT NULL,
    ward_id INT NOT NULL,
    height_admission REAL,
    weight_admission REAL,
    weight_discharge REAL,
    hospital_admit_time TEXT NOT NULL,
    hospital_admission_source TEXT NOT NULL,
    unit_admit_time TEXT NOT NULL,
    unit_discharge_time TEXT,
    hospital_discharge_time TEXT,
    hospital_discharge_status TEXT
);

DROP TABLE IF EXISTS condition;
CREATE TABLE condition
(
    condition_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    condition_name TEXT NOT NULL,
    condition_time TEXT NOT NULL,
    icd9_code TEXT,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS treatment;
CREATE TABLE treatment
(
    treatment_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    treatment_name TEXT NOT NULL,
    treatment_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS lab;
CREATE TABLE lab
(
    lab_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    lab_name TEXT NOT NULL,
    lab_result REAL NOT NULL,
    lab_result_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS prescription;
CREATE TABLE prescription
(
    prescription_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    drug_name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    administration_route TEXT NOT NULL,
    medication_start_time TEXT,
    medication_stop_time TEXT,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS cost;
CREATE TABLE cost
(
    cost_id INT NOT NULL PRIMARY KEY,
    hosp_id INT NOT NULL,
    unit_id INT NOT NULL,
    event_type TEXT NOT NULL,
    event_id INT NOT NULL,
    cost_time TEXT NOT NULL,
    cost_amount REAL NOT NULL,
    FOREIGN KEY(hosp_id) REFERENCES icupatient(hosp_id)
);

DROP TABLE IF EXISTS allergy_reaction;
CREATE TABLE allergy_reaction
(
    allergy_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    drug_name TEXT,
    allergy_name TEXT NOT NULL,
    allergy_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS fluid_balance;
CREATE TABLE fluid_balance
(
    fluid_balance_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    fluid_path TEXT NOT NULL,
    fluid_label TEXT NOT NULL,
    fluid_value_numeric REAL NOT NULL,
    fluid_balance_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS microbiology;
CREATE TABLE microbiology
(
    microbiology_id INT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    culture_site TEXT NOT NULL,
    organism TEXT NOT NULL,
    culture_taken_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS vital_signs;
CREATE TABLE vital_signs
(
    vital_sign_id BIGINT NOT NULL PRIMARY KEY,
    unit_id INT NOT NULL,
    temperature REAL,
    sao2 INT,
    heart_rate INT,
    respiration_rate INT,
    systolic_bp INT,
    diastolic_bp INT,
    mean_bp INT,
    vital_time TEXT NOT NULL,
    FOREIGN KEY(unit_id) REFERENCES icupatient(unit_id)
);

DROP TABLE IF EXISTS hospital;
CREATE TABLE hospital
(
    hospital_id INT NOT NULL PRIMARY KEY,
    bed_capacity_category TEXT,
    teaching_status TEXT,
    region TEXT
);