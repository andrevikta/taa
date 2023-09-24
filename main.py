"""--------------------------------------------
Muat Library untuk Naive Bayes dan PCA
--------------------------------------------"""
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.core.window import Window

"""--------------------------------------------
Muat dataset dan persiapkan fitur serta target
--------------------------------------------"""
# Ganti dengan nama file dataset yang Anda miliki
filename = r"south_africa_heart_disease.csv"
dataframe = read_csv(filename, delimiter=";")

# Pastikan nama kolom-kolom yang diinginkan benar-benar ada dalam dataset
desired_columns = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "famhist",
    "typea",
    "obesity",
    "alcohol",
    "age",
    "target",
]

missing_columns = [col for col in desired_columns if col not in dataframe.columns]
if missing_columns:
    raise ValueError(
        f"Kolom-kolom berikut tidak ditemukan dalam dataset: {missing_columns}"
    )

# Mengganti 'Present' dengan 1 dan 'Absent' dengan 0 pada kolom 'famhist'
dataframe["famhist"] = dataframe["famhist"].replace({"Present": 1, "Absent": 0})

# Pisahkan dataset menjadi fitur (X) dan target (Y)
X = dataframe[
    desired_columns[:-1]
]  # Mengambil semua kolom kecuali 'target' sebagai fitur
Y = dataframe[desired_columns[-1]]  # Kolom 'target' sebagai target

# Proses scaling menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lakukan PCA untuk ekstraksi fit   1`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````  ur pada data yang telah di-scaling
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Pisahkan dataset menjadi set pelatihan dan pengujian
test_size = 0.1
seed = 42
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pca, Y, test_size=test_size, random_state=seed
)

# Buat dan latih model klasifikasi Naive Bayes
model = GaussianNB()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)


"""-----------------------------------
Receive data user input & formatting it
-----------------------------------"""


def user_report(sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age):
    sbp = float(sbp)
    tobacco = float(tobacco)
    ldl = float(ldl)
    adiposity = float(adiposity)
    famhist = float(famhist)
    typea = float(typea)
    obesity = float(obesity)
    alcohol = float(alcohol)
    age = float(age)

    user_report_data = {
        "sbp": sbp,
        "tobacco": tobacco,
        "ldl": ldl,
        "adiposity": adiposity,
        "famhist": famhist,
        "typea": typea,
        "obesity": obesity,
        "alcohol": alcohol,
        "age": age,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


"""------------------------
Kivy properties's interface
------------------------"""
Builder_string = """
ScreenManager:
    Main:

<Main>:
    name : 'main'
    MDLabel:
        text: 'Deteksi Dini Penyakit Gagal Jantung'
        halign: 'center'
        pos_hint: {'center_y':0.98}
        font_size: '24sp'
        color: '#5B2160'
        size_hint_y: None  # Tentukan tinggi elemen
        height: self.texture_size[1]  # Gunakan tinggi teks sebagai tinggi elemen

    GridLayout:
        cols: 2
        padding: '10dp'
        spacing: '2dp'

        MDLabel:
            text: 'sbp (tekanan darah sistolik pasien)'
        MDTextField:
            id: input_1
            hint_text: '(0 - 300)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'tobacco (kadar konsumsi rokok pasien)'
        MDTextField:
            id: input_2
            hint_text: '(0.0 - 30.0)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'ldl (kadar kolesterol ldl pasien)'
        MDTextField:
            id: input_3
            hint_text: '(0.0 - 30.0)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'adiposity (kadar adipositas pasien)'
        MDTextField:
            id: input_4
            hint_text: '(0.0 - 50.0)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'famhist (apakah pasien memiliki riwayat keluarga penyakit jantung atau tidak. Jika ada = 1 | tidak ada = 0)'
        MDTextField:
            id: input_5
            hint_text: '(0 atau 1)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'typea (kadar perilaku tipe A Pasien)'
        MDTextField:
            id: input_6
            hint_text: '(0 - 100)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'obesity (kadar obesitas pasien)'
        MDTextField:
            id: input_7
            hint_text: '(0.0 - 50)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'alcohol (kadar konsumsi alkohol pasien)'
        MDTextField:
            id: input_8
            hint_text: '(0.0 - 100.0)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'age (usia pasien)'
        MDTextField:
            id: input_9
            hint_text: '(15 - 70)'
            width: 100
            size_hint_x: None

        MDLabel:
            text: 'Prediction Result'
        MDLabel:
            pos_hint: {'center_y':0.2}
            halign: 'center'
            text: ''
            id: output_text_not
            theme_text_color: "Custom"
            text_color: '#3B8BFF'
        MDLabel:
            pos_hint: {'center_y':0.2}
            halign: 'center'
            text: ''
            id: output_text_sick
            theme_text_color: "Custom"
            text_color: '#EF0F0F'

        MDRaisedButton:
            pos_hint: {'center_x':0.5}
            text: 'Predict'
            on_press: app.predict()

"""

"""----------------------------------
Class for call properties's interface
----------------------------------"""


class Main(Screen):
    pass


sm = ScreenManager()
sm.add_widget(Main(name="main"))

"""------------------------------
Class main for execute our system
------------------------------"""


class MainApp(MDApp):
    def build(self):
        self.help_string = Builder.load_string(Builder_string)
        return self.help_string

    def predict(self):
        input_1 = self.help_string.get_screen("main").ids.input_1.text
        input_2 = self.help_string.get_screen("main").ids.input_2.text
        input_3 = self.help_string.get_screen("main").ids.input_3.text
        input_4 = self.help_string.get_screen("main").ids.input_4.text
        input_5 = self.help_string.get_screen("main").ids.input_5.text
        input_6 = self.help_string.get_screen("main").ids.input_6.text
        input_7 = self.help_string.get_screen("main").ids.input_7.text
        input_8 = self.help_string.get_screen("main").ids.input_8.text
        input_9 = self.help_string.get_screen("main").ids.input_9.text

        user_result = user_report(
            input_1,
            input_2,
            input_3,
            input_4,
            input_5,
            input_6,
            input_7,
            input_8,
            input_9,
        )

        user_result = model.predict(user_result)

        output = ""
        self.help_string.get_screen("main").ids.output_text_not.text = ""
        self.help_string.get_screen("main").ids.output_text_sick.text = ""
        if user_result[0] == 0:
            output = "Anda Tidak Berpotensi Terkena Penyakit Gagal Jantung"
            self.help_string.get_screen("main").ids.output_text_not.text = output
        else:
            output = "Anda Berpotensi Terkena Penyakit Gagal Jantung"
            self.help_string.get_screen("main").ids.output_text_sick.text = output
        print(output)
        print("Accuracy:  ", result * 100.0)


"""-------------
Run our programm
-------------"""
MainApp().run()
