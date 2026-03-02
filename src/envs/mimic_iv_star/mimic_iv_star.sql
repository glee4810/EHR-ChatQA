DROP TABLE IF EXISTS demographics;
CREATE TABLE demographics 
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    gender TEXT NOT NULL,
    dateofbirth TEXT NOT NULL,
    dateofdeath TEXT
);

DROP TABLE IF EXISTS hospitaladmissions;
CREATE TABLE hospitaladmissions
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    admitdatetime TEXT NOT NULL,
    dischargedatetime TEXT,
    admissiontype TEXT NOT NULL,
    admitsource TEXT NOT NULL,
    dischargedestination TEXT,
    insurancetype TEXT NOT NULL,
    language TEXT,
    maritalstatus TEXT,
    age INT NOT NULL,
    FOREIGN KEY(patientid) REFERENCES demographics(patientid)
);

DROP TABLE IF EXISTS diagnosiscodes;
CREATE TABLE diagnosiscodes
(
    recordid INT NOT NULL PRIMARY KEY,
    icdcode TEXT NOT NULL,
    codeversion INT NOT NULL,
    description TEXT NOT NULL
);

DROP TABLE IF EXISTS procedurecodes;
CREATE TABLE procedurecodes 
(
    recordid INT NOT NULL PRIMARY KEY,
    icdcode TEXT NOT NULL,
    codeversion INT NOT NULL,
    description TEXT NOT NULL
);

DROP TABLE IF EXISTS labtesttypes;
CREATE TABLE labtesttypes 
(
    recordid INT NOT NULL PRIMARY KEY,
    itemcode INT NOT NULL,
    itemname TEXT
);

DROP TABLE IF EXISTS clinicalitemtypes;
CREATE TABLE clinicalitemtypes 
(
    recordid INT NOT NULL PRIMARY KEY,
    itemcode INT NOT NULL,
    itemname TEXT NOT NULL,
    abbreviation TEXT NOT NULL,
    itemtype TEXT NOT NULL
);

DROP TABLE IF EXISTS admissiondiagnoses;
CREATE TABLE admissiondiagnoses
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icdcode TEXT NOT NULL,
    codeversion INT NOT NULL,
    recordeddatetime TEXT NOT NULL,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(codeversion, icdcode) REFERENCES diagnosiscodes(codeversion, icdcode)
);

DROP TABLE IF EXISTS admissionprocedures;
CREATE TABLE admissionprocedures
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icdcode TEXT NOT NULL,
    codeversion INT NOT NULL,
    recordeddatetime TEXT NOT NULL,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(codeversion, icdcode) REFERENCES procedurecodes(codeversion, icdcode)
);

DROP TABLE IF EXISTS labresults;
CREATE TABLE labresults
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    itemcode INT NOT NULL,
    resultdatetime TEXT,
    resultvalue DOUBLE PRECISION,
    resultunit TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(itemcode) REFERENCES labtesttypes(itemcode)
);

DROP TABLE IF EXISTS medicationorders;
CREATE TABLE medicationorders
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    startdatetime TEXT NOT NULL,
    enddatetime TEXT,
    medicationname TEXT NOT NULL,
    dosevalue TEXT NOT NULL,
    doseunit TEXT NOT NULL,
    administrationroute TEXT NOT NULL,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid)
);

DROP TABLE IF EXISTS costrecords;
CREATE TABLE costrecords
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    eventtype TEXT NOT NULL,
    costid INT NOT NULL,
    costdatetime TEXT NOT NULL,
    costamount DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(costid) REFERENCES admissiondiagnoses(recordid),
    FOREIGN KEY(costid) REFERENCES admissionprocedures(recordid),
    FOREIGN KEY(costid) REFERENCES labresults(recordid),
    FOREIGN KEY(costid) REFERENCES medicationorders(recordid)
);

DROP TABLE IF EXISTS clinicalevents;
CREATE TABLE clinicalevents
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icuadmissionid INT NOT NULL,
    itemcode INT NOT NULL,
    recordeddatetime TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(icuadmissionid) REFERENCES icuepisodes(icuadmissionid),
    FOREIGN KEY(itemcode) REFERENCES clinicalitemtypes(itemcode)
);

DROP TABLE IF EXISTS intakerecords;
CREATE TABLE intakerecords
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icuadmissionid INT NOT NULL,
    startdatetime TEXT NOT NULL,
    itemcode INT NOT NULL,
    totalvolume DOUBLE PRECISION,
    volumeunit TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(icuadmissionid) REFERENCES icuepisodes(icuadmissionid),
    FOREIGN KEY(itemcode) REFERENCES clinicalitemtypes(itemcode)
);

DROP TABLE IF EXISTS outputrecords;
CREATE TABLE outputrecords
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icuadmissionid INT NOT NULL,
    recordeddatetime TEXT NOT NULL,
    itemcode INT NOT NULL,
    volume DOUBLE PRECISION,
    volumeunit TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid),
    FOREIGN KEY(icuadmissionid) REFERENCES icuepisodes(icuadmissionid),
    FOREIGN KEY(itemcode) REFERENCES clinicalitemtypes(itemcode)
);

DROP TABLE IF EXISTS microbiologyresults;
CREATE TABLE microbiologyresults
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    collecteddatetime TEXT NOT NULL,
    specimentype TEXT,
    testname TEXT,
    organismname TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid)
);

DROP TABLE IF EXISTS icuepisodes;
CREATE TABLE icuepisodes
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    icuadmissionid INT NOT NULL,
    initialcareunit TEXT NOT NULL,
    finalcareunit TEXT NOT NULL,
    admitdatetime TEXT NOT NULL,
    dischargedatetime TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid)
);

DROP TABLE IF EXISTS patienttransfers;
CREATE TABLE patienttransfers
(
    recordid INT NOT NULL PRIMARY KEY,
    patientid INT NOT NULL,
    admissionid INT NOT NULL,
    transferid INT NOT NULL,
    transfertype TEXT NOT NULL,
    careunit TEXT,
    transferindatetime TEXT NOT NULL,
    transferoutdatetime TEXT,
    FOREIGN KEY(admissionid) REFERENCES hospitaladmissions(admissionid)
);