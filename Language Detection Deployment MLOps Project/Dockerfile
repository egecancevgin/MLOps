FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

# $ docker build -t language-detection-app
# $ docker run -p 80:80 language-detection-app
# 0.0.0.0/docs
# $ git init
# ! add .gitignore and type venv
# $ heroku login
# $ touch heroku.yml
# $ git add heroku.yml
# $ git commit -m "Add heroku.yml"
# $ heroku create language-detection-app-12
# $ heroku git:remote language-detection-app-12
# $ heroku stack:set container
# $ git branch -M main
# $ git push heroku main
# Test in postman .../predict