The id files are from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)

Install the MIMIC-III database with PostgreSQL following this [instruction](https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/)

Generate the train/valid/test sets using `src/util/mimiciii_data_processing.py`.
(Config the connection to PostgreSQL in Line 139)

