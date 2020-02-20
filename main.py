from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

def modelleri_olustur(girdi_sayisi, cikti_sayisi, hucre_sayisi):
    #egitim kodlayıcısı
    kodlayici_girdileri = Input(shape = (None, girdi_sayisi))
    kodlayici = LSTM(hucre_sayisi, return_state=True)
    kodlayici_ciktilari, durum_h, durum_c = encoder(kodlayici_girdileri)
    kodlayici_durumlari = [durum_h, durum_c]
    #kod çözücü
    kod_cozucu_girdileri = Input(shape = (None, cikti_sayisi))
    kod_cozucu_lstm = LSTM(hucre_sayisi, return_sequences=True, return_state=True)
    kod_cozucu_ciktilari, _, _ = decoder_lstm(kodlayici_girdileri, initial_state=kodlayici_durumlari)
    kod_cozucu_dense = Dense(cikti_sayisi, activation='softmax')
    kod_cozucu_ciktilari = decoder_dense(kod_cozucu_ciktilari)
    model = Model([kodlayici_girdileri, kod_cozucu_girdileri], kod_cozucu_ciktilari)

    #çıkarım inference kodlayıcısı
    kodlayici_model = Model(kodlayici,kodlayici_durumlari)
    #çıkarım inference çözücüsü
    kod_cozucu_durum_girdi_h = Input(shape=(hucre_sayisi,))
    kod_cozucu_durum_girdi_c = Input(shape=(hucre_sayisi,))
    kod_cozucu_durumlari_girdiler = [kod_cozucu_durum_girdi_h,kod_cozucu_durum_girdi_c]

    kod_cozucu_ciktilari, durum_h, durum_c = decoder_lstm(kod_cozucu_girdileri, initial_state=kod_cozucu_durumlari_girdiler)
    kod_cozucu_durumlari = [durum_c, durum_h]
    kod_cozucu_ciktilari = decoder_dense(kod_cozucu_ciktilari)
    kod_cozucu_model = Model([kod_cozucu_girdileri] + kod_cozucu_durumlari_girdiler, [kod_cozucu_ciktilari] + kod_cozucu_durumlari)

    #seq2seq modelinin derlenmesi
    egitim_modeli, kodlayici_model, kod_cozucu_model = modelleri_olustur(45, 45, 64)
    egitim_modeli.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
    egitim_modeli.fit([X1, X2], Y, epochs = 5)

    #tüm modelleri döndür
    return egitim_modeli, kodlayici_model, kod_cozucu_model

