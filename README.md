Steps to Run the Code

1:make an vitual ennvironment
python -m venv venv

2:activate the virtual environment source
venv/bin/activate

3:install the required packages
pip install -r requirements.txt

4:intall
python -m spacy download en_core_web_sm

5:Make Directory (folder) name
"Uploaded_Resumes"

5:now after the venv environment is being made we need to change some files code and add some files for the latest version compatibility of spacy

6: in the path \venv\Lib\site-packages\pyresparser ADD one file name
"config.cfg"

7: the code inside the config.cfg
    [paths]
    model = en_core_web_sm
    
8: now we can make the sql connection;
make a DB named "cv" and inside the code change the password to your SQL DB(workbench) password;

10: now we can run the code and "streamlit run App.py"
 
 
