import json
import requests
import streamlit as st

st.title("Tarductor") 

oracion_en = st.text_input("Texto en ingles", value="")

if st.button('Taducir', type="primary"):
    if oracion_en is None:
        st.error("No hay texto a traducir")
    else:
        #st.text("Traduciendo...")
        data_json = {
            "oreacion_en": oracion_en
        }
        response = requests.post("http://127.0.0.1:8000/model/translate", data=json.dumps(data_json))
        
        data = response.json()
        
        traduccion=data["traduccion"]

        def lista_a_cadena(lista):
            lista_filtrada = [palabra for palabra in lista if palabra != '<eos>']
            cadena = ' '.join(lista_filtrada)
            return cadena
        
        resultado = lista_a_cadena(traduccion)
        st.text("Traduccion Español:")
        st.text(resultado)
        

  

        
   
