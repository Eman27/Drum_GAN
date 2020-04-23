# Springboard

The Problem
  When practicing an instrument such as guitar, the act of training alone can feel tedious and boring. Practicing scales, chords and patterns to a metronome can become monotonous does not spark the same joy most feel when playing as part of a group. One can play along to songs to simulate this experience but it does not allow the player to be creative in their approach to the song. They are just following along, making sure they are hitting the correct notes at the right moments. The goal of this project is to generate drum tracks for the player so that they have something unique and original to play alongside. The purpose of these tracks is to allow the player to apply their theory and delve into the creative aspects of playing such as song writing and improvision.
  
Local:
   - It is recommended to set up the project inside a virtual enviroment to keep the dependencies separated.
      - Python
      - Conda
   - Activate your virtual enviroment.
   - Install dependencies by running pip install -r requirements.txt
   - Start up the server using python /drum_gan/app.py
   - Visit http://0.0.0.0:5000/ to explore and test

Docker:
  - Mac:
    - git clone https://github.com/Eman27/Springboard.git
    - cd Springboard/drum_gan
    - docker build -sudo docker build -t drumgan .
    - docker sudo docker run -p 5000:5000 drumgan
  - Windows:
    - git clone https://github.com/Eman27/Springboard.git
    - cd Springboard/drum_gan
    - docker build -sudo docker build -t drumgan .
    - docker sudo docker run -p 5000:5000 drumgan
  - Linux:
    - git clone https://github.com/Eman27/Springboard.git
    - cd Springboard/drum_gan
    - docker build -sudo docker build -t drumgan .
    - docker sudo docker run -p 5000:5000 drumgan
  - Cloud:
    - Create a VM in Azure, GCP or AWS
    - git clone https://github.com/Eman27/Springboard.git
    - cd Springboard/drum_gan
    - docker build sudo docker build -t drumgan
    - docker sudo docker run -p 5000:5000 drumgan

- Then go to http://0.0.0.0:5000/
 
Deployment:
- The deloyment can be found at the following link:
- http://9bbc55aa.ngrok.io

Running API Calls:
- Training:
  - URL
    - /train
  - Method
    - POST
  - Required Data Params:
    - For specific genre:
      - {'genre':'rock'}
    - For full dataset training:
      - leave blank
      
- Generating
  - URL
    - /gen
  - Method
    - POST
  - Required Data Params:
    - For specific genre:
      - {genre:'rock'}
    - For full dataset training:
      - leave blank
