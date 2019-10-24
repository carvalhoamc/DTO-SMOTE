import glob
import os

arquivos = glob.glob('*.pdf')
for f in arquivos:
	os.system('pdfcrop ' + f + ' ' + 'crop_'+f )