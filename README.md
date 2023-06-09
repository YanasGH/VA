# Visualization Tool GAStech
This repository contains the code for the interactive dashboard for GAStech of Group 3. Below you can find the instructions on how to launch it locally.

## A. Requirements

* Python==3.9 or Python==3.10

## B. Dependencies

First prepare and activate your python environment (e.g. `conda`). Then, use the requirments.txt from the folder.  
Using pip:
```bash
$ pip install -r requirements.txt

```

## C. How to run
1. Clone the git repository (unzip the zip file containing the dashboard).
2. Open anaconda prompt (terminal) and set current directory to the path name of the unzipped folder ("VA").
3. In anaconda prompt (terminal), enter
```bash
$ python gastech.py

```
4. The application should now be running on your local system (after a couple of seconds).
5. To view the dashboard, open your preferred web browser and enter in the address bar `http://127.0.0.1:3000/`.

(In case the dashboard displays overlapping components try opening the dashboard in a smaller venster instead of full screen mode.)

## D. Additional files
* the assets folder contains all a css file (used for styling the dashboard), 20 html files (10 html files containing a chord diagram per department and 10 html files containing a chord diagram per sentiment), 12 woff files and a "generator_config.txt"" file for the font, one json file used to style the chord diagrams and one png file containing the word cloud
* "home.py" (in the "apps" folder) contains the home page of the dashboard
* data/articles folder that contains the txt files with news articles
* "EmployeeRecords.xlsx" contains the employee records
* "email_headers.csv" contains the email correspondences between employees over two working weeks
* "email_classification.csv" contains the classified email headers; it is generated by running classification_emails.ipynb in google colab (https://colab.research.google.com/drive/1R-n4NiRPUdSEvF8-mgQWIDR_DtBEEBCR?usp=sharing)
* "department_chord_graph_main.py" is the file used to generate 10 department chord diagram (one for each day available in the data); run "python department_chord_graph_main.py" if you wish to regenerate the diagrams
* "sentiment_chord_graph_main.py" is the file used to generate 10 sentiment chord diagram (one for each day available in the data); run "python sentiment_chord_graph_main.py" if you wish to regenerate the diagrams
* "wordcloud_generator.py" is the file used to generate the word cloud; run "python wordcloud_generator.py" if you wish to regenerate it




