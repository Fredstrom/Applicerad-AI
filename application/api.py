from nn_model.train_and_predict import predict_image
from settings import settings

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid

# Initialize FastAPI and loads all data-folders to use.
app = FastAPI()
templates = Jinja2Templates(directory='application/templates')
app.mount("/static", StaticFiles(directory="application/static"), name="static")


# Endpoints
@app.get("/")
async def home(request: Request):
    """ Loads an HTML-file containing a form to use for choice of image to upload"""
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/prediction")
async def upload_file(request: Request, file: UploadFile = File(...), location: str = Form()):
    """
    Takes an image input and saves the image to static/[api_folder], then makes a prediction
    on the image using our model, saving the results in a dictionary and passes it to a HTML
    template for visualisation.

    Parameters:
    class request:
        request (class from fastapi)
    str file:
        Image from form at home.html
    str location:
        String from form, containing the location the image was taken from

    returns HTMLTemplate(name, dict)
    """

    location = location
    file.filename = f'{str(uuid.uuid4())[:8]}-{location}.jpg'
    image = await file.read()

    with open(f'{settings["api_folder"]}{file.filename}', 'wb') as f:
        f.write(image)
        label, confidence = predict_image(f'{settings["api_folder"]}', file.filename, location)

    return templates.TemplateResponse("predict.html", {"request": request,
                                                       "label": label,
                                                       "confidence": confidence,
                                                       "image": file.filename,
                                                       "location": location})
