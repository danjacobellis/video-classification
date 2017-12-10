from flask import send_file, Flask, current_app, request, jsonify
import io
import os
import sys
import json
import base64
import logging
import matplotlib
matplotlib.use('Agg')
import subprocess
import numpy as np
import glob
import requests
from extractor import Extractor
from wordcloud import WordCloud
from multiprocessing import Pool
from keras.models import load_model
from werkzeug.datastructures import MultiDict

app = Flask(__name__)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

@app.route('/', methods=['POST', 'GET'])
def pred():
    data = ""
    found = False

    try:
        data = request.get_json()['data']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400
    print(data)
    ### begin predictions    
    input_video = data
    fpv = 40
#     if not os.path.exists('frames-' + str(fpv)):
#         os.makedirs('frames-' + str(fpv))        
#     frame_dest_name = 'frames-' + str(fpv) + '/frame' + '-%03d.jpg'
    
    #extract features
    if 'feature_extractor' not in globals():
        global feature_extractor            
        feature_extractor = Extractor(frames_per_video=fpv)        
    frame_dir = 'frames-' + str(fpv)
    sequence, timing = feature_extractor.extract(input_video,frame_dir)
    
    # make predictions
    if not 'trained_lstm_model' in globals():    
        global trained_lstm_model
#        file_id = '0BxnNE6IbgiIJUTNtUVpiNjFIeVU'
        file_id = '1iV9jOuvsy9060_U8n6Fikvo_xHMw1g-b'

#        destination = 'lstm-features.062-1.015.hdf5'
        destination = 'ucf-40-gru-stage1.hdf5'
        if os.path.isfile(destination):
            print('Found model: ',destination)
        else: 
            print('Downloading ' + destination + '...')
            download_file_from_google_drive(file_id, destination)
        print('loading model...')
        trained_lstm_model = load_model(destination)
    X = np.zeros([1,np.shape(sequence)[0], np.shape(sequence)[1]])
    X[0,:,:] = sequence
    preds = trained_lstm_model.predict(X)
    top5 = np.argsort(-preds)[0][0:5]
    
    class_names = ["Apply Eye Makeup","Apply Lipstick","Archery","Baby Crawling","Balance Beam","Band Marching","Baseball Pitch","Basketball Shooting","Basketball Dunk","Bench Press","Biking","Billiards Shot","Blow Dry Hair","Blowing Candles","Body Weight Squats","Bowling","Boxing Punching Bag","Boxing Speed Bag","Breaststroke","Brushing Teeth","Clean and Jerk","Cliff Diving","Cricket Bowling","Cricket Shot","Cutting In Kitchen","Diving","Drumming","Fencing","Field Hockey Penalty","Floor Gymnastics","Frisbee Catch","Front Crawl","Golf Swing","Haircut","Hammer Throw","Hammering","Handstand Pushups","Handstand Walking","Head Massage","High Jump","Horse Race","Horse Riding","Hula Hoop","Ice Dancing","Javelin Throw","Juggling Balls","Jump Rope","Jumping Jack","Kayaking","Knitting","Long Jump","Lunges","Military Parade","Mixing Batter","Mopping Floor","Nun chucks","Parallel Bars","Pizza Tossing","Playing Guitar","Playing Piano","Playing Tabla","Playing Violin","Playing Cello","Playing Daf","Playing Dhol","Playing Flute","Playing Sitar","Pole Vault","Pommel Horse","Pull Ups","Punch","Push Ups","Rafting","Rock Climbing Indoor","Rope Climbing","Rowing","Salsa Spins","Shaving Beard","Shotput","Skate Boarding","Skiing","Skijet","Sky Diving","Soccer Juggling","Soccer Penalty","Still Rings","Sumo Wrestling","Surfing","Swing","Table Tennis Shot","Tai Chi","Tennis Swing","Throw Discus","Trampoline Jumping","Typing","Uneven Bars","Volleyball Spiking","Walking with a dog","Wall Pushups","Writing On Board","Yo Yo"]        
    prob = preds[0][top5]
    names = [class_names[i] for i in top5]
    
    ### end of prediction
    
    ### make variable names work with old version of code
    classes = class_names
    word_count = [count * 10**15 for count in preds[0]]
    
    first_prob = prob[0]
    second_prob = prob[1]
    third_prob = prob[2]
    fourth_prob = prob[3]
    fifth_prob = prob[4]    
    
    top1 = names[0]
    top2 = names[1]
    top3 = names[2]
    top4 = names[3]
    top5 = names[4]    
    
    def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
        return("hsl(230,100%%, %d%%)" % np.random.randint(49,51))
    
    wordcloud = WordCloud(width = 500, height = 281, min_font_size=4, 
                relative_scaling=0.65, background_color='white')
    frequencies = dict()
    for i in range(100):
        frequencies[classes[i]] = int(word_count[i] * (10**15))
    wordcloud.generate_from_frequencies(frequencies)
    wordcloud.recolor(color_func = grey_color_func)
    wordcloud.to_file('wordcloud.png')

    with open("wordcloud.png", "rb") as image_file:
        wordcloud_image = base64.b64encode(image_file.read())

    predictions = {
                    'label1': top1, 'label2': top2, 'label3': top3, 'label4': top4, 'label5': top5,
                    'prob1': str(first_prob * 100), 'prob2': str(second_prob * 100), 'prob3': str(third_prob * 100), 
                    'prob4': str(fourth_prob * 100), 'prob5': str(fifth_prob * 100), 'wordcloud': wordcloud_image
                  }
    preds = predictions.copy()
    preds.pop('wordcloud')

    current_app.logger.info('Predictions: %s', preds)

    return jsonify(predictions=predictions)

#def init_model():
#    global trained_lstm_model
#    file_id = '0BxnNE6IbgiIJUTNtUVpiNjFIeVU'
#    destination = 'tools/lstm-features.062-1.015.hdf5'
#    if os.path.isfile(destination):
#        print('Found model: ',destination)
#    else: 
#        print('Downloading ' + destination + '...')
#        download_file_from_google_drive(file_id, destination)
#    print('loading model...')
#    trained_lstm_model = load_model(destination)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
#    init_model()
