import unittest
import os
import sqlite3
from PIL import Image
from UI import app, dbconn
from flask import request, session
from werkzeug.datastructures import FileStorage

os.environ['DATABASE_URL'] = 'sqlite:///deepfake.db'

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.app.testing = True
    
    def tearDown(self):
        self.app_context.pop()
        self.app = None
        self.app_context = None

    def test_app(self):
        self.assertIsNotNone(self.app)

    def test_dbconn(self):
        self.assertIsNotNone(dbconn)

class TestURL(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.app.testing = True
        self.client = self.app.test_client()
    
    def tearDown(self):
        self.app_context.pop()
        self.app = None
        self.app_context = None
        self.client = None
    
    def test_main(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_home(self):
        response = self.client.get('/home')
        self.assertEqual(response.status_code, 200)
    
    def test_index(self):
        response = self.client.get('/index')
        self.assertEqual(response.status_code, 200)
    
    def test_login(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
    
    def test_signup(self):
        response = self.client.get('/signup')
        self.assertEqual(response.status_code, 200)
    
    def test_logout(self):
        response = self.client.get('/logout')
        self.assertEqual(response.status_code, 200)
    
    def test_profile(self):
        response = self.client.get('/profile')
        self.assertEqual(response.status_code, 200)
    
    def test_predict(self):
        response = self.client.get('/predict/*')
        self.assertEqual(response.status_code, 302) # redirect to home page
    
    def test_result(self):
        response = self.client.get('/get_result/*')
        self.assertEqual(response.status_code, 302) # redirect to home page
    
    def test_model(self):
        response = self.client.get('/model')
        self.assertEqual(response.status_code, 200)
    
    def test_change_model(self):
        response = self.client.get('/change_model/dfdc.h5')
        self.assertEqual(response.status_code, 302) # redirect to model page

class TestSystemComponent(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.app.testing = True
        self.client = self.app.test_client()
    
    def tearDown(self):
        self.app_context.pop()
        self.app = None
        self.app_context = None
        self.client = None

    
    
    def test_imageupload(self):
        with open('custom_data/Fake/fake_1.png', 'rb') as f:
            image_data = f.read()
        response = self.client.post('/', data={'image': image_data}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

    def test_imageupload_fail(self):
        with self.client.session_transaction() as sess:
            sess['model'] = 'dfdc'
        response = self.client.post('/predict/uploads')
        self.assertEqual(response.status_code, 400)
    
    def test_imagesave(self):
        with self.client.session_transaction() as sess:
            sess['model'] = 'dfdc'
        response = self.client.post('/get_result/uploads', data={'saveimage': 'on'}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fake', response.data)
        self.assertTrue(os.path.exists('static/permittedimg/uploadedimage_0.png'))
    
    def test_modelchange(self):
        with self.client.session_transaction() as sess:
            response = self.client.get('/change_model/stylegan.h5')
            self.assertEqual(sess['model'], 'stylegan')
            self.assertEqual(response.status_code, 302)
    
    def test_register(self):
        response = self.client.post('/signup', data={
            'username': 'testuser',
            'email': 'testuser@uwe.com',
            'password': 'testpass'}, follow_redirects=True)
        self.assertEqual(response.status_code, 302) #redirect to login page
        self.assertIn(b'Login', response.data)
    
    def test_existinguser(self):
        response = self.client.post('/signup', data={
            'username': 'testuser',
            'email': 'testuser@uwe.com',
            'password': 'testpass'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'User already exists', response.data)
    
    def test_login(self):
        response = self.client.post('/login', data={
            'email': 'testuser@uwe.com',
            'password': 'testpass'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_login_fail(self):
        response = self.client.post('/login', data={
            'email': 'notexistinguser',
            'password': 'testpass'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_modelupload(self):
        with self.client.session_transaction() as sess:
            with open('models/dfdc.h5', 'rb') as f:
                model_data = f.read()
            response = self.client.post('/model', data={'modelfile': model_data}, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(sess['model'], 'dfdc')
    
    def test_modelpage(self):
        with self.client.session_transaction() as session:
            session['logged_in'] = True
 
            response = self.client.get('/model')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Model', response.data)
            self.assertIn(b'Accuracy', response.data)
            self.assertIn(b'Loss', response.data)

if __name__ == '__main__':
    unittest.main()