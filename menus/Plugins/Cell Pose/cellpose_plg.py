import numpy as np
import pandas as pd
from imagepy.core.engine import Simple
from imagepy import IPy
from imagepy.core import ImagePlus

class Plugin(Simple):
	title = 'Cell Pose'
	note = ['all']
	para = {'model':'cyto', 'cytoplasm':0, 'nucleus':0, 'flow':False, 'diams':False, 'slice':False}
	view = [(list, 'model', ['cyto', 'nuclei'], str, 'model', ''),
			(list, 'cytoplasm', [0,1,2,3], int, 'cytoplasm', 'channel'),
			(list, 'nucleus', [0,1,2,3], int, 'nucleus', 'channel'),
			(bool, 'flow', 'show color flow'),
			(bool, 'diams', 'show diams tabel'),
			(bool, 'slice', 'slice')]

	def run(self, ips, imgs, para = None):
		print('aaa')
		import mxnet as mx
		from cellpose import models, utils
		print('bbb')

		if not para['slice']: imgs = [ips.img]
		imgs = [i.reshape((i.shape+(-1,))[:3]) for i in imgs]
		device = mx.gpu() if utils.use_gpu() else mx.cpu()
		model = models.Cellpose(device, model_type=para['model'])
		channels = [para['cytoplasm'], para['nucleus']]
		self.setValue = lambda x: self.progress(x, 100)
		masks, flows, styles, diams = model.eval(
			imgs, rescale=None, channels=channels, progress=self)
		IPy.show_img(masks, ips.title+'-cp_mask')
		if para['flow']: IPy.show_img([i[0] for i in flows], ips.title+'-cp_flow')
		if para['diams']: 
			IPy.show_table(pd.DataFrame({'diams': diams}), ips.title+'-cp_diams')