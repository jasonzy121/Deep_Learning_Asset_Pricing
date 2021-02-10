import numpy as np

from src.utils import deco_print

class FirmChar:
	def __init__(self):
		self._category = ['Past Returns', 'Investment', 'Profitability', 'Intangibles', 'Value', 'Trading Frictions']
		self._category2variables = {
			'Past Returns': ['r2_1', 'r12_2', 'r12_7', 'r36_13', 'ST_REV', 'LT_Rev'],
			'Investment': ['Investment', 'NOA', 'DPI2A', 'NI'],
			'Profitability': ['PROF', 'ATO', 'CTO', 'FC2Y', 'OP', 'PM', 'RNA', 'ROA', 'ROE', 'SGA2S', 'D2A'],
			'Intangibles': ['AC', 'OA', 'OL', 'PCM'],
			'Value': ['A2ME', 'BEME', 'C', 'CF', 'CF2P', 'D2P', 'E2P', 'Q', 'S2P', 'Lev'],
			'Trading Frictions': ['AT', 'Beta', 'IdioVol', 'LME', 'LTurnover', 'MktBeta', 'Rel2High', 'Resid_Var', 'Spread', 'SUV', 'Variance']
		}
		self._variable2category = {}
		for category in self._category:
			for var in self._category2variables[category]:
				self._variable2category[var] = category
		self._category2color = {
			'Past Returns': 'red', 
			'Investment': 'green', 
			'Profitability': 'grey', 
			'Intangibles': 'magenta', 
			'Value': 'purple', 
			'Trading Frictions': 'orange'
		}
		self._color2category = {value:key for key, value in self._category2color.items()}

	def getColorLabelMap(self):
		return {var: self._category2color[self._variable2category[var]] for var in self._variable2category}

class DataInRamInputLayer:
	def __init__(self, 
				pathIndividualFeature, 
				pathMacroFeature=None,
				macroIdx=None, 
				meanMacroFeature=None, 
				stdMacroFeature=None, 
				normalizeMacroFeature=True):
		self._UNK = -99.99
		self._load_individual_feature(pathIndividualFeature)
		self._load_macro_feature(pathMacroFeature, macroIdx, meanMacroFeature, stdMacroFeature, normalizeMacroFeature)
		self._firm_char = FirmChar()

	def _create_var_idx_associations(self, varList):
		idx2var = {idx:var for idx, var in enumerate(varList)}
		var2idx = {var:idx for idx, var in enumerate(varList)}
		return idx2var, var2idx

	def _load_individual_feature(self, pathIndividualFeature):
		tmp = np.load(pathIndividualFeature)
		data = tmp['data']
		
		### Data Stored Here		
		self._return = data[:,:,0]
		self._individualFeature = data[:,:,1:]
		self._mask = (self._return != self._UNK)

		### Dictionary
		self._idx2date, self._date2idx = self._create_var_idx_associations(tmp['date'])
		self._idx2var, self._var2idx = self._create_var_idx_associations(tmp['variable'][1:])
		self._dateCount, self._permnoCount, self._varCount = data.shape
		self._varCount -= 1

	def _load_macro_feature(self, pathMacroFeature, macroIdx=None, meanMacroFeature=None, stdMacroFeature=None, normalizeMacroFeature=True):
		if pathMacroFeature is None:
			self._macroFeature = np.empty(shape=[self._dateCount, 0])
			self._meanMacroFeature = None
			self._stdMacroFeature = None
		else:
			tmp = np.load(pathMacroFeature)
			if macroIdx is None:
				macro_idx = np.arange(len(tmp['variable']))
			elif macroIdx == 'all':
				macro_idx = np.arange(len(tmp['variable']))
			elif type(macroIdx) is list:
				macro_idx = np.sort(np.array(macroIdx, dtype=int))
			elif macroIdx == '178':
				macro_idx = np.sort(np.concatenate((np.arange(124), np.arange(284, 338))))
			else:
				macro_idx = []
				deco_print('WARNING: macroIdx not supported! Use no macro variables. ')
			self._macroFeature = tmp['data'][:,macro_idx]
			if normalizeMacroFeature:
				if meanMacroFeature is None or stdMacroFeature is None:
					self._meanMacroFeature = self._macroFeature.mean(axis=0)
					self._stdMacroFeature = self._macroFeature.std(axis=0)
				else:
					self._meanMacroFeature = meanMacroFeature
					self._stdMacroFeature = stdMacroFeature
				self._macroFeature -= self._meanMacroFeature
				self._macroFeature /= self._stdMacroFeature
			else:
				self._meanMacroFeature = None
				self._stdMacroFeature = None

			self._idx2var_macro, self._var2idx_macro = self._create_var_idx_associations(tmp['variable'][macro_idx])
			self._varCount_macro = self._macroFeature.shape[1]

	def getDateCountList(self):
		return np.sum(self._mask, axis=0)

	def getDateList(self):
		return [self._idx2date[i] for i in range(self._dateCount)]

	def getIndividualFeatureList(self):
		return [self._idx2var[i] for i in range(self._varCount)]

	def getDateByIdx(self, idx):
		return self._idx2date[idx]

	def getIndividualFeatureByIdx(self, idx):
		return self._idx2var[idx]

	def getIdxByIndividualFeature(self, var):
		return self._var2idx[var]

	def getMacroFeatureList(self):
		return [self._idx2var_macro[i] for i in range(self._varCount_macro)]

	def getMacroFeatureByIdx(self, idx):
		return self._idx2var_macro[idx]

	def getFeatureByIdx(self, idx):
		if idx < self._varCount:
			return self.getIndividualFeatureByIdx(idx)
		else:
			return self.getMacroFeatureByIdx(idx - self._varCount)

	def getMacroFeatureMeanStd(self):
		return self._meanMacroFeature, self._stdMacroFeature

	def getIndividualFeatureColarLabelMap(self):
		return self._firm_char.getColorLabelMap(), self._firm_char._color2category

	def iterateOneEpoch(self, subEpoch=False):
		if subEpoch:
			for _ in range(subEpoch):
				yield self._macroFeature, self._individualFeature, self._return, self._mask
		else:
			yield self._macroFeature, self._individualFeature, self._return, self._mask