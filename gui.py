from kivy.config import Config
# Set window size to simulate mobile resolution
Config.set('graphics', 'width', '360')
Config.set('graphics', 'height', '640')

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.graphics import Rectangle, Color
from detect import predict_diabetic_retinopathy

kivy.require('2.0.0')

class RetinopathyApp(App):

    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Set background color for BoxLayout
        with layout.canvas.before:
            Color(0.9, 0.9, 0.9, 1)  # Set the color (light yellow)
            self.rect = Rectangle(size=layout.size, pos=layout.pos)
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # Model Accuracy Label and TextInput
        model_accuracy_label = Label(text="Model Accuracy:", size_hint=(1, 0.1), color=(0, 0, 0, 1))  # Black text color
        layout.add_widget(model_accuracy_label)
        
        self.model_accuracy_input = TextInput(readonly=True, size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5})
        layout.add_widget(self.model_accuracy_input)

        # Image Panel
        self.image = Image(source='placeholder.png', size_hint=(0.6, 0.4), pos_hint={'center_x': 0.5})
        layout.add_widget(self.image)

        # Upload Button
        self.upload_button = Button(text="Upload Image", size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5}, background_normal= '', background_color=(0.1, 0.2, 1, 1))  # Light blue background
        self.upload_button.bind(on_press=self.upload_image)
        layout.add_widget(self.upload_button)

        # Predict Button
        self.predict_button = Button(text="Predict", size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5}, background_normal= '', background_color=(0.0, 0.5, 0.0, 1), disabled=True)  # Start with the button disabled
        self.predict_button.bind(on_press=self.predict)
        layout.add_widget(self.predict_button)

        # Severity Level Label and TextInput
        severity_label = Label(text="Severity of Diabetic Retinopathy:", size_hint=(1, 0.1), color=(0, 0, 0, 1))  # Black text color
        layout.add_widget(severity_label)
        
        self.severity_input = TextInput(readonly=True, size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5})
        layout.add_widget(self.severity_input)

        # Clear Button
        clear_button = Button(text="Clear", size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5}, background_normal= '', background_color=(1, 1, 0, 1), color=(0, 0, 0, 1))  # Dark yellow background
        clear_button.bind(on_press=self.clear_fields)  # Bind the button to the clear_fields method
        layout.add_widget(clear_button)

        # Exit Button
        exit_button = Button(text="Exit", size_hint=(0.6, 0.1), pos_hint={'center_x': 0.5}, background_normal= '', background_color=(1, 0.1, 0.1, 1))  # Red background
        exit_button.bind(on_press=self.stop)  # Bind to the stop method to close the app
        layout.add_widget(exit_button)

        return layout
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def upload_image(self, instance):
        # Create a FileChooser widget
        filechooser = FileChooserIconView(filters=['*.png', '*.jpg', '*.jpeg'])

        # Create a BoxLayout to contain the FileChooser and the Submit button
        box = BoxLayout(orientation='vertical')
        box.add_widget(filechooser)

        # Create a Submit button
        submit_button = Button(text="Submit", size_hint_y=None, height=44)
        submit_button.bind(on_press=lambda instance: self.update_image_and_close_popup(filechooser, popup))
        box.add_widget(submit_button)

        # Create a Popup widget to contain the BoxLayout
        popup = Popup(title="Select an Image", content=box, size_hint=(0.9, 0.9))

        # Open the popup
        popup.open()

    def update_image_and_close_popup(self, filechooser, popup):
        # Check if any file is selected
        if filechooser.selection:
            # Update the Image widget's source to the selected file
            self.image.source = filechooser.selection[0]
            self.predict_button.disabled = False  # Enable the Predict button

            # Close the popup
            popup.dismiss()

    def predict(self, instance):
        with open('accuracy.txt', 'r') as file:
            accuracy = float(file.readline().strip())

        # Convert the value to a percentage format
        formatted_accuracy = "{:.2f}%".format(accuracy * 100)
        # Assuming self.image.source contains the path to the uploaded image
        uploaded_image_path = self.image.source
        
        # Predict using the model
        result = predict_diabetic_retinopathy(uploaded_image_path)
        
        # Update the relevant text field or label with the result
        # Read the accuracy value from the text file


        # Display the formatted value in the model accuracy text field
        self.model_accuracy_input.text = formatted_accuracy
        
    
        self.severity_input.text = result

    def clear_fields(self, instance):
        self.image.source = ''  # Reset to placeholder or blank image
        self.model_accuracy_input.text = ''  # Clear the text input
        self.severity_input.text = ''  # Clear the text input
        self.predict_button.disabled = True  # Disable the Predict button

    def exit_app(self, instance):
        exit()


if __name__ == "__main__":
    RetinopathyApp().run()
