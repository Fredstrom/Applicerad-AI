from nn_model.train_and_predict import predict_image
from settings import settings
import uuid

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Takes an image input and saves the image to static/[api_folder], then makes a prediction
    on the image using our model, saving the results in a dictionary and passes it to a HTML
    template for visualisation.

    :param request: default (Request-object)
    :param file: Image from home.html - or any other form.
    :return: HTMLTemplate(name, dict)
    """

    file.filename = f'{uuid.uuid4()}.jpg'
    image = await file.read()

    with open(f'{settings["api_folder"]}{file.filename}', 'wb') as f:
        f.write(image)
        label = predict_image(f'{settings["api_folder"]}', file.filename)

    return templates.TemplateResponse("predict.html", {"request": request,
                                                       "label": label,
                                                       "image": file.filename})
