from django.http import HttpResponse, JsonResponse
from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.hashers import check_password, make_password
from django.contrib.auth import logout as auth_logout, authenticate,login 
from django.views.decorators.csrf import csrf_exempt
from .models import ISLGif, SignLanguageLetter
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from .utils.final import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp


def home(request):
    print(request)
    return render(request, 'index.html')


def about(request):
    return render(request, 'aboutus.html')

def features(request):
    return render(request, 'service.html')

def contactpage(request):
    return render(request, 'con.html')

def back(request):
    return render(request, 'mainnew.html')


from django.contrib.auth.models import User

def signup_view(request):
    print('goodydm')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')

        print(f"Received: {username}, {password}, {email}")

        # Checking if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return render(request, 'registration.html')

        # Checking if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email address is already in use.")
            return render(request, 'registration.html')

        # Hash the password
        hashed_password = make_password(password)

        try:
            # Create a new user
            # print('inside try')
            user = User(username=username, password=hashed_password, email=email)
            user.save()
            # print('User saved successfully.')

            # Optional: Add a message for successful registration
            messages.success(request, 'Your account has been created successfully!')
            return redirect('login')
            # return render(request, 'login.html')  # Redirect to the login page after successful registration
        except Exception as e:
            # Log the exception and show a generic error message
            print(f"Error saving user: {e}")
            messages.error(request, "An error occurred while creating your account. Please try again later.")
            return render(request, 'registration.html')
    return render(request, 'registration.html')

def login_views(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)
        print('auth hai')
        if user is not None:
            print('user ke ander')
            login(request, user) 
            messages.success(request, 'You have been successfully logged in.')
            return render(request, 'index.html')
        else:
            messages.error(request, 'Invalid username or password. Please try again.')

    return render(request, 'login.html')


def logout(request):
    auth_logout(request)  
    return render(request, 'index.html') 



def tryyy(request):
    return render(request, 'tryy.html')

def videocheck(request):
    return render(request, 'camera.html')


@csrf_exempt
def speech_to_text(request):
    if request.method == 'POST':
        try:
            print("try k ander")
            data = json.loads(request.body)
            recognized_text = data.get('text', '').lower()
            print(f"Received Text: {recognized_text}")

            # Check if recognized text matches a GIF (from MongoDB)
            gif_record = ISLGif.objects.filter(phrase=recognized_text).first()
            print("kuch gif_record mila h ", gif_record)
            if gif_record:
                print("yahi zole h")
                return JsonResponse({'gif_url': gif_record.gif_url})

            # Check if recognized text matches a letter and return corresponding image (from MongoDB)
            image_urls = []
            for c in recognized_text:
                print("images par aa gya wo")
                image_record = SignLanguageLetter.objects.filter(letter=c).first()
                print("image record print karwa diya")
                if image_record:
                    image_urls.append(image_record.image_url)

            if image_urls:
                print("sabse alg issue ")
                return JsonResponse({'image_urls': image_urls})

            return JsonResponse({'message': 'Text not recognized'})
        
        except Exception as e:
            return JsonResponse({'message': f"Error processing text: {str(e)}"})

    return JsonResponse({'message': 'Invalid request method'})


from django.shortcuts import render
from .models import ISLGif, SignLanguageLetter
import json
from django.http import JsonResponse

@csrf_exempt
def text_to_sign(request):
    if request.method == 'POST':
        try:
            recognized_text = request.POST.get('inputText', '').lower()
            print(f"Received Text: {recognized_text}")

            # Check if recognized text matches a GIF (from ISLGif model)
            gif_record = ISLGif.objects.filter(phrase=recognized_text).first()
            print("Found GIF record:", gif_record)
            
            if gif_record:
                gif_url = gif_record.gif_url
                return render(request, 'texttosign.html', {'gif_url': gif_url})

            # Check if recognized text matches a letter and return corresponding image (from SignLanguageLetter model)
            image_urls = []
            for c in recognized_text:
                image_record = SignLanguageLetter.objects.filter(letter=c).first()
                if image_record:
                    image_urls.append(image_record.image_url)

            if image_urls:
                return render(request, 'texttosign.html', {'image_urls': image_urls})

            return render(request, 'texttosign.html', {'message': 'Text not recognized'})

        except Exception as e:
            return render(request, 'texttosign.html', {'message': f"Error processing text: {str(e)}"})

    return render(request, 'texttosign.html', {'message': 'Invalid request method'})



# Load the model
model = load_model('/home/t/Pictures/collegeproject/myproject/myapp/utils/actionnarayanprabhu.h5')

# Define the actions
actions = np.array(['Hello', 'Thanks', 'I like you', "Home", "Beautiful"])
mp_holistic = mp.solutions.holistic

# Define colors for visualization
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (19, 117, 245), (16, 100, 245)]


# Define the prob_viz function for visualization
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    
    # Ensure there are enough colors for the actions
    if len(res) > len(colors):
        raise ValueError(f"Not enough colors for {len(res)} actions. You need {len(res)} colors.")
    
    # Loop through the probabilities for each action
    for num, prob in enumerate(res):
        # If prob is a list or array, access its value (if necessary)
        if isinstance(prob, (list, np.ndarray)):
            prob = prob[0]  # Or whatever index you need, for example prob[0]
        
        # Draw rectangle based on probability (ensure prob is a scalar)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        
        # Add text with the action name
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame


def capture_video(request):
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    threshold = 0.8

    # Use MediaPipe for holistic detection
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Extract keypoints and make predictions
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Visualization and prediction result
                image = prob_viz(res, actions, image, colors)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return JsonResponse({"message": "Camera feed processed successfully."})


