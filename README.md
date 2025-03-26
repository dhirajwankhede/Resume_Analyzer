# Steps to Run the Code
### 1:make an vitual ennvironment 
    python -m venv venv
### 2:activate the virtual environment source 
    venv/bin/activate
### 3:install the required packages 
    pip install -r requirements.txt
### 4:intall 
    python -m spacy download en_core_web_sm
### 5:Make Directory (folder) name 
    "Uploaded_Resumes"
### 5:now after the venv environment is being made we need to change some files code and add some files for the latest version compatibility of spacy
### 6: in the path \venv\Lib\site-packages\pyresparser ADD one file name 
    "config.cfg"
### 7: the code inside the config.cfg
        [paths]
        model = en_core_web_sm
### 8:the code inside utils.py , the function def extract_name (nlp_text,matcher) should be;
'''
    Helper function to extract name from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    '''
    pattern = cs.NAME_PATTERN

    matcher.add('NAME', [pattern])


    matches = matcher(nlp_text)

    for _, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            return span.text

### 9: now we can make the sql connection;
# make a DB named "cv" and inside the code change the password to your SQL DB(workbench) password;
### 10: now we can run the code and "streamlit run App.py"

