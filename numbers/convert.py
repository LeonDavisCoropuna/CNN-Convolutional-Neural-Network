from PIL import Image

try:
    resample_filter = Image.Resampling.LANCZOS
except AttributeError:
    resample_filter = Image.ANTIALIAS  # para versiones anteriores a Pillow 10

def convertir_a_bn_28x28(ruta_entrada, ruta_salida, umbral=128):
    imagen = Image.open(ruta_entrada)
    imagen = imagen.convert('L')
    imagen = imagen.resize((28, 28), resample_filter)
    imagen_bn = imagen.point(lambda x: 255 if x > umbral else 0, mode='1')
    imagen_bn = imagen_bn.convert('L')
    imagen_bn.save(ruta_salida)
    print(f'Imagen guardada en {ruta_salida}')

convertir_a_bn_28x28('Captura desde 2025-05-19 17-08-57.png', 'Captura desde 2025-05-19 17-08-57.png')
