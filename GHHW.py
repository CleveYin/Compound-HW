# -*- coding: utf-8 -*-
# @Author: ethan
# @Date:   2022-12-31 17:18:51
# @Last Modified by:   ethan
# @Last Modified time: 2023-01-05 18:26:06

import os
import cdsapi
c = cdsapi.Client()
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import pygmt
import cv2
from threading import Thread
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# 创建文件夹
def CreateDir(path_1):
	if os.path.exists(path_1):
		pass
	else:
		os.makedirs(path_1)

# 下载ERA5-HEAT数据
def DownloadData(outPath_1):
	for year_1 in range(1972, 2022):
		outPath_2 = os.path.join(outPath_1, 'ERA5-HEAT_' + str(year_1) + '.zip')
		if os.path.exists(outPath_2):
			print('ERA5-HEAT_' + str(year_1) + ' exists!')
		else:
			c.retrieve(
			    'derived-utci-historical',
			    {
			        'version': '1_1',
			        'format': 'zip',
			        'variable': 'universal_thermal_climate_index',
			        'product_type': 'consolidated_dataset',
			        'year': year_1,
			        'month': [
			            '01', '02', '03',
			            '04', '05', '06',
			            '07', '08', '09',
			            '10', '11', '12',
			        ],
			        'day': [
			            '01', '02', '03',
			            '04', '05', '06',
			            '07', '08', '09',
			            '10', '11', '12',
			            '13', '14', '15',
			            '16', '17', '18',
			            '19', '20', '21',
			            '22', '23', '24',
			            '25', '26', '27',
			            '28', '29', '30',
			            '31',
			        ],
			    },
			    outPath_2)
			print('ERA5-HEAT_' + str(year_1) + ' is done!')

# 重命名20210101-20210428的数据
def RenameFile(inPath_1):
	inPath_2 = os.path.join(inPath_1, 'ERA5-HEAT_2021')
	filenameList_1 = sorted(os.listdir(inPath_2))
	for filename_1 in filenameList_1:
		inPath_3 = os.path.join(inPath_2, filename_1)
		outPath_1 = os.path.join(inPath_2, filename_1.replace('v1.0', 'v1.1'))
		os.rename(inPath_3, outPath_1)
		print(filename_1 + ' is done!')

# 逐小时计算白天/夜间判断矩阵
def SunTime(timeArray_1, outPath_1):
	outPath_2 = os.path.join(outPath_1, '2_SunTime')
	CreateDir(outPath_2)
	for time_1 in timeArray_1:
		outPath_3 = os.path.join(outPath_2, 'suntime_' + time_1.strftime('%Y-%m-%d_%H') + '.nc')
		if os.path.exists(outPath_3):
			print(time_1.strftime('%Y-%m-%d_%H') + ' exists!')
		else:
			# 生成全球逐小时白天/夜间分区图片，并保存
			fig_1 = pygmt.Figure()
			fig_1.coast(region = 'd', projection = None, land = 'white', water = 'white')
			fig_1.solar(terminator = 'day_night', terminator_datetime = time_1, fill = 'black', pen = '0.5p')
			outPath_4 = os.path.join(outPath_2, 'suntime_' + time_1.strftime('%Y-%m-%d_%H') + '.png')
			fig_1.savefig(outPath_4, dpi = 244)
			
			# 读取图片，并将图片由RGB格式转为灰度值，并删除图片
			fig_2 = cv2.imread(outPath_4)
			sunArray_1 = cv2.cvtColor(fig_2, cv2.COLOR_RGB2GRAY)
			sunArray_2 = np.where(sunArray_1 >= 127.5, 1, 0)
			os.remove(outPath_4)
			
			# 将数据保存为NC格式
			latArray_1 = np.arange(90, -60.25, -0.25)
			lonArray_1 = np.arange(-180, 180, 0.25)
			sunArray_3 = sunArray_2[1: len(latArray_1) + 1, 1: len(lonArray_1) + 1]
			dataset_1 = xr.Dataset({'utci': (['lat', 'lon'], sunArray_3)}, coords = {'lat': (['lat'], latArray_1), 'lon': (['lon'], lonArray_1)})
			dataset_1.to_netcdf(outPath_3)
			print(time_1.strftime('%Y-%m-%d_%H') + ' is done!')

# 计算日白天/夜间平均温度
def DailyMean(inPath_1, inPath_2, outPath_1):
	for year_1 in range(1972, 2022):
		outPath_2 = os.path.join(outPath_1, '3_DailyMean', 'day', 'ERA5-HEAT_' + str(year_1))
		outPath_3 = os.path.join(outPath_1, '3_DailyMean', 'night', 'ERA5-HEAT_' + str(year_1))
		CreateDir(outPath_2)
		CreateDir(outPath_3)
		dateArray_1 = pd.date_range(start = str(year_1) + '-01-01', end = str(year_1) + '-12-31', freq = 'D')
		for date_1 in dateArray_1:
			outPath_4 = os.path.join(outPath_2, 'ECMWF_utci_' + date_1.strftime('%Y%m%d') + '_day_v1.1_con.nc')
			outPath_5 = os.path.join(outPath_3, 'ECMWF_utci_' + date_1.strftime('%Y%m%d') + '_night_v1.1_con.nc')
			if os.path.exists(outPath_4) & os.path.exists(outPath_5):
				print(date_1.strftime('%Y%m%d') + ' exists!')
			else:
				inPath_3 = os.path.join(inPath_1, 'ERA5-HEAT_' + str(year_1), 'ECMWF_utci_' + date_1.strftime('%Y%m%d') + '_v1.1_con.nc')
				try:
					datasetDay_1 = xr.open_dataset(inPath_3)
				except Exception as e:
					print('ERA5-HEAT_' + str(year_1), 'ECMWF_utci_' + date_1.strftime('%Y%m%d') + '_v1.1_con.nc does not exists!')
					continue
				hourArray_1 = datasetDay_1['time'].data
				datasetDayList_1 = []
				datasetNightList_1 = []
				for hour_1 in hourArray_1:
					datasetHour_1 = datasetDay_1.sel(time = hour_1)
					inPath_4 = os.path.join(inPath_2, 'suntime_2000-' + str(hour_1)[5: 7] + '-' + str(hour_1)[8: 10] + '_' + str(hour_1)[11: 13] + '.nc')
					datasetSuntime_1 = xr.open_dataset(inPath_4)
					datasetHourDay_1 = xr.where(datasetSuntime_1, datasetHour_1, np.nan)
					datasetHourNight_1 = xr.where(1 - datasetSuntime_1, datasetHour_1, np.nan)
					# plt.imshow(datasetHourDay_1['utci'])
					# plt.show()
					datasetDayList_1.append(datasetHourDay_1)
					datasetNightList_1.append(datasetHourNight_1)
				datasetDayDay_1 = xr.concat(datasetDayList_1, dim = 'time')
				datasetDayNight_1 = xr.concat(datasetNightList_1, dim = 'time')
				datasetDayDayMean_1 = datasetDayDay_1.mean(dim = 'time')
				datasetDayNightMean_1 = datasetDayNight_1.mean(dim = 'time')
				datasetDayDayMean_1.to_netcdf(outPath_4)
				datasetDayNightMean_1.to_netcdf(outPath_5)
				print(date_1.strftime('%Y%m%d') + ' is done!')

# 计算百分位数阈值（日尺度）
def PercThre(inPath_1, tag_1, timeArray_1, days_1, percList_1):
	inPath_2 = os.path.join(inPath_1, '3_DailyMean', tag_1)

	# 打开一个模板数据
	inPath_3 = os.path.join(inPath_2, os.listdir(inPath_2)[0])
	inPath_4 = os.path.join(inPath_3, os.listdir(inPath_3)[0])
	datasetDemo_1 = xr.open_dataset(inPath_4)
	latArray_1 = datasetDemo_1['lat'].data
	lonArray_1 = datasetDemo_1['lon'].data

	outPath_1 = os.path.join(inPath_1, '4_PercThre', tag_1)
	CreateDir(outPath_1)
	for time_1 in timeArray_1:
		outPath_2 = os.path.join(outPath_1, 'precthre_' + str(percList_1[0]) + '-' + str(percList_1[-1]) + '_' + tag_1 + '_' + time_1.strftime('%Y-%m-%d') + '.nc')
		if os.path.exists(outPath_2):
			print(time_1.strftime('%Y-%m-%d') + ' exists!')
		else:
			timeGap_1 = time_1 - datetime.datetime(2000, 1, 1)	# 与第一天相差的天数
			timeGap_2 = datetime.datetime(2000, 12, 31) - time_1 # 与最后一天相差的天数
			timeRange_1 = []	# 该日时间窗口列表
			if (timeGap_1.days >= days_1) & (timeGap_2.days >= days_1):
				timeRange_1 = [time_1 - datetime.timedelta(days = days_1), time_1 + datetime.timedelta(days = days_1)]
			elif timeGap_1.days < days_1:
				timeRange_1 = [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 1) + datetime.timedelta(days = days_1 * 2)]
			elif timeGap_2.days < days_1:
				timeRange_1 = [datetime.datetime(2000, 12, 31) - datetime.timedelta(days = days_1 * 2), datetime.datetime(2000, 12, 31)]
			dataList_1 = []
			for year_1 in range(1972, 2022):
				timeRange_2 = []
				try:
					timeRange_2 = [datetime.datetime(year_1, timeRange_1[0].month, timeRange_1[0].day), datetime.datetime(year_1, timeRange_1[1].month, timeRange_1[1].day)]
				except Exception as e:
					time_2 = datetime.datetime(year_1, time_1.month, time_1.day)
					timeRange_2 = [time_2 - datetime.timedelta(days = days_1), time_2 + datetime.timedelta(days = days_1)]
				for time_3 in timeRange_2:
					inPath_5 = os.path.join(inPath_2, 'ERA5-HEAT_' + str(time_3.year), 'ECMWF_utci_' + time_3.strftime('%Y%m%d') + '_' + tag_1 + '_v1.1_con.nc')
					try:
						datasetDay_1 = xr.open_dataset(inPath_5)
						dataArray_1 = datasetDay_1['utci'].data
						if len(dataArray_1.shape) == 2:
							dataList_1.append(dataArray_1)
						else:
							dataList_1.append(dataArray_1[:, :, 0])
					except Exception as e:
						print(e)
			dataArray_2 = np.dstack(dataList_1)
			dataArray_3 = np.full((len(percList_1), len(latArray_1), len(lonArray_1)), np.nan)
			index_1 = 0
			for perc_1 in percList_1:
				dataArray_3[index_1, :, :] = np.quantile(dataArray_2, perc_1, axis = 2)
				index_1 = index_1 + 1
			datasetDay_3 = xr.Dataset({'utci': (['perc', 'lat', 'lon'], dataArray_3)}, coords = {'perc': (['perc'], percList_1), 'lat': (['lat'], latArray_1), 'lon': (['lon'], lonArray_1)})
			datasetDay_3.to_netcdf(outPath_2)
			print(time_1.strftime('%Y-%m-%d') + ' is done!')

# 计算日内热浪
def HeatWaveHour(datasetDayStart_1, datasetDayEnd_1, datasetTagHours_1, datasetTsumHours_1, datasetFreqHours_1, datasetDuraHours_1, datasetTsumHours_2, datasetTemp_1, datasetTempThre_1, tempThre_1, hourThre_1):
	datasetTagHours_1 = xr.where(datasetDayStart_1, 0, datasetTagHours_1)	# 将第一个小时的标记初始化为0
	datasetTsumHours_1 = xr.where(datasetDayStart_1, 0, datasetTsumHours_1)	# 将第一个小时的累计温度初始化为0，以计算一天内的累计温度
	datasetFreqHours_1 = xr.where(datasetDayStart_1, 0, datasetFreqHours_1)	# 将第一个小时的热浪频次初始化为0，以计算一天内的热浪频次
	datasetDuraHours_1 = xr.where(datasetDayStart_1, 0, datasetDuraHours_1)	# 将第一个小时的热浪持续时间初始化为0，以计算一天内的热浪持续时间
	datasetTsumHours_2 = xr.where(datasetDayStart_1, 0, datasetTsumHours_2) # 将第一个小时的累计温度初始化为0，以计算一天内的累计温度

	datasetBoolHours_1 = xr.where((datasetTemp_1 >= tempThre_1 + 273.15) & (datasetTemp_1 >= datasetTempThre_1), 1, 0) # 温度高于阈值的像元标记为1
	datasetTagHours_1 = xr.where(datasetBoolHours_1, datasetTagHours_1 + 1, datasetTagHours_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1
	datasetTsumHours_1 = xr.where(datasetBoolHours_1, datasetTsumHours_1 + datasetTemp_1, datasetTsumHours_1)	# 标记为1的像元，即温度高于阈值的像元，累计温度数加1

	datasetCondHours_1 = ((1 - datasetBoolHours_1) | (datasetBoolHours_1 & datasetDayEnd_1)) & (datasetTagHours_1 >= hourThre_1)
	datasetFreqHours_1 = xr.where(datasetCondHours_1, datasetFreqHours_1 + 1, datasetFreqHours_1)	
	datasetDuraHours_1 = xr.where(datasetCondHours_1, datasetDuraHours_1 + datasetTagHours_1, datasetDuraHours_1)
	datasetTsumHours_2 = xr.where(datasetCondHours_1, datasetTsumHours_2 + datasetTsumHours_1, datasetTsumHours_2)

	datasetTagHours_1 = xr.where((1 - datasetBoolHours_1) | datasetDayEnd_1 , 0, datasetTagHours_1)
	datasetTsumHours_1 = xr.where((1 - datasetBoolHours_1) | datasetDayEnd_1, 0, datasetTsumHours_1)

	return datasetTagHours_1, datasetTsumHours_1, datasetFreqHours_1, datasetDuraHours_1, datasetTsumHours_2

# 计算白天/夜间高温热浪
def HeatWaveSubday(inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, threList_1, outPath_1):
	outPath_2 = os.path.join(outPath_1, '5_HeatWave', threList_1[0])
	CreateDir(outPath_2)
	for year_1 in range(timeRange_1[0], timeRange_1[1]):
		timeArray_1 = pd.date_range(start = str(year_1) + '-' + timeRange_1[2], end = str(year_1) + '-' + timeRange_1[3])
		outPath_3 = os.path.join(outPath_2, 'hwr_' + threList_1[0] + '_' + str(threList_1[1]) + '_' + str(threList_1[2]) + '_' + str(threList_1[3]) + '_' + str(threList_1[4]) + '_' + timeArray_1[0].strftime('%Y%m%d') + '-' + timeArray_1[-1].strftime('%Y%m%d') + '.nc')
		if os.path.exists(outPath_3):
			print(timeArray_1[0].strftime('%Y%m%d') + '-' + timeArray_1[-1].strftime('%Y%m%d') + ' exists.')
		else:
			datasetTagHours_1 = 0
			datasetTsumHours_1 = 0
			datasetFreqHours_1 = 0 # 频次
			datasetDuraHours_1 = 0 # 持续时间
			datasetTsumHours_2 = 0 # 持续时间

			datasetTagDays_1 = 0
			datasetHsumDays_1 = 0
			datasetTsumDays_1 = 0
			datasetHsumDays_2 = 0
			datasetTsumDays_2 = 0
			datasetFreqDays_1 = 0 # 频次
			datasetDuraDays_1 = 0 # 持续时间
			datasetStartDays_1 = 0
			datasetEndDays_1 = 0

			days_2 = timeArray_1[-1] - timeArray_1[0]
			for time_1 in timeArray_1:
				print(threList_1[0] + ': ' + time_1.strftime('%Y%m%d'))
				inPath_4 = os.path.join(inPath_1, 'ERA5-HEAT_' + str(year_1), 'ECMWF_utci_' + time_1.strftime('%Y%m%d') + '_v1.1_con.nc')
				inPath_5 = os.path.join(inPath_3, threList_1[0], 'precthre_0.8-0.95_' + threList_1[0] + '_2000-' + time_1.strftime('%m-%d') + '.nc')
				try:
					datasetTemp_1 = xr.open_dataset(inPath_4)
					datasetTempThre_1 = xr.open_dataset(inPath_5)
				except Exception as e:
					print(time_1.strftime('%Y%m%d') + ' does not exists!')	
				datasetTemp_2 = datasetTemp_1.where(datasetMask_1)	# 该小时陆地区域温度
				datasetTempThre_2 = datasetTempThre_1.sel(perc = threList_1[2])
				timeArray_2 = datasetTemp_2['time'].data
				for time_2 in timeArray_2:
					time_2 = pd.to_datetime(time_2)
					datasetTemp_3 = datasetTemp_2.sel(time = time_2)

					inPath_6 = os.path.join(inPath_2, 'suntime_2000-' + (time_2 - datetime.timedelta(hours = 1)).strftime('%m-%d_%H') + '.nc')	# 前一小时的白天/夜间情况
					inPath_7 = os.path.join(inPath_2, 'suntime_2000-' + time_2.strftime('%m-%d_%H') + '.nc')	# 该小时的白天/夜间情况
					inPath_8 = os.path.join(inPath_2, 'suntime_2000-' + (time_2 + datetime.timedelta(hours = 1)).strftime('%m-%d_%H') + '.nc')	# 后一小时的白天/夜间情况
					datasetSuntime_1 = xr.open_dataset(inPath_6)	# 前一小时的白天/夜间情况
					datasetSuntime_2 = xr.open_dataset(inPath_7)	# 该小时的白天/夜间情况
					datasetSuntime_3 = xr.open_dataset(inPath_8)	# 后一小时的白天/夜间情况
					if threList_1[0] == 'night':
						datasetSuntime_1 = 1 - datasetSuntime_1
						datasetSuntime_2 = 1 - datasetSuntime_2
						datasetSuntime_3 = 1 - datasetSuntime_3
					datasetDayStart_1 = xr.where((1 - datasetSuntime_1) & datasetSuntime_2, 1, 0)	# 将白天/夜间的第一个小时像元标记为1，作为一天的开始
					datasetDayEnd_1 = xr.where(datasetSuntime_2 & (1 - datasetSuntime_3), 1, 0)	# 将白天/夜间的最后一个小时像元标记为1，作为一天的结束

					datasetTemp_4 = xr.where(datasetSuntime_2, datasetTemp_3, np.nan)	# 该小时白天/夜间温度
					datasetTagHours_1, datasetTsumHours_1, datasetFreqHours_1, datasetDuraHours_1, datasetTsumHours_2 = HeatWaveHour(datasetDayStart_1, datasetDayEnd_1, datasetTagHours_1, \
						datasetTsumHours_1, datasetFreqHours_1, datasetDuraHours_1, datasetTsumHours_2, datasetTemp_4, datasetTempThre_2, threList_1[1], threList_1[3])

					datasetBoolDays_1 = xr.where(datasetDayEnd_1 & (datasetFreqHours_1 >= 1), 1, 0)
					datasetTagDays_1 = xr.where(datasetBoolDays_1, datasetTagDays_1 + 1, datasetTagDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1
					datasetHsumDays_1 = xr.where(datasetBoolDays_1, datasetHsumDays_1 + datasetDuraHours_1, datasetHsumDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1
					datasetTsumDays_1 = xr.where(datasetBoolDays_1, datasetTsumDays_1 + datasetTsumHours_2 / datasetDuraHours_1, datasetTsumDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1

					datasetCondDays_1 = (((1 - datasetBoolDays_1) & datasetDayEnd_1) | (datasetBoolDays_1 & (time_1.year == timeArray_1[-1].year) & (time_1.month == timeArray_1[-1].month) & (time_1.day == timeArray_1[-1].day))) & (datasetTagDays_1 >= threList_1[3])
					datasetFreqDays_1 = xr.where(datasetCondDays_1, datasetFreqDays_1 + 1, datasetFreqDays_1)	
					datasetDuraDays_1 = xr.where(datasetCondDays_1, datasetDuraDays_1 + datasetTagDays_1, datasetDuraDays_1)
					datasetHsumDays_2 = xr.where(datasetCondDays_1, datasetHsumDays_2 + datasetHsumDays_1, datasetHsumDays_2)
					datasetTsumDays_2 = xr.where(datasetCondDays_1, datasetTsumDays_2 + datasetTsumDays_1, datasetTsumDays_2)
					datasetStartDays_1 = xr.where(datasetCondDays_1 & (datasetStartDays_1 == 0), int(time_1.strftime('%j')) - datasetTagDays_1, datasetStartDays_1)
					datasetEndDays_1 = xr.where(datasetCondDays_1, int(time_1.strftime('%j')), datasetEndDays_1)

					datasetCondDays_2 = ((1 - datasetBoolDays_1) & datasetDayEnd_1) | (time_1.year == timeArray_1[-1].year) & (time_1.month == timeArray_1[-1].month) & (time_1.day == timeArray_1[-1].day)
					datasetTagDays_1 = xr.where(datasetCondDays_2, 0, datasetTagDays_1)	# 标记为0的像元，即温度低于阈值的像元，本次高温过程中断，累计小时数初始化为0
					datasetHsumDays_1 = xr.where(datasetCondDays_2, 0, datasetHsumDays_1)
					datasetTsumDays_1 = xr.where(datasetCondDays_2, 0, datasetTsumDays_1)

			datasetFreqDays_1 = datasetFreqDays_1.where(datasetFreqDays_1 > 0)
			datasetDuraDays_1 = datasetDuraDays_1.where(datasetFreqDays_1 > 0)
			datasetHavgDays_1 = datasetHsumDays_2 / datasetDuraDays_1
			datasetTavgDays_1 = datasetTsumDays_2 / datasetDuraDays_1
			datasetStartDays_1 = datasetStartDays_1.where(datasetFreqDays_1 > 0)
			datasetEndDays_1 = datasetEndDays_1.where(datasetFreqDays_1 > 0)

			datasetFreqDays_1 = datasetFreqDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetDuraDays_1 = datasetDuraDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetHavgDays_1 = datasetHavgDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetTavgDays_1 = datasetTavgDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetStartDays_1 = datasetStartDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetEndDays_1 = datasetEndDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))

			datasetFreqDays_1 = datasetFreqDays_1.rename({'utci': 'HWF'})
			datasetDuraDays_1 = datasetDuraDays_1.rename({'utci': 'HWD'})
			datasetHavgDays_1 = datasetHavgDays_1.rename({'utci': 'HWH'})
			datasetTavgDays_1 = datasetTavgDays_1.rename({'utci': 'HWT'})
			datasetStartDays_1 = datasetStartDays_1.rename({'utci': 'HWS'})
			datasetEndDays_1 = datasetEndDays_1.rename({'utci': 'HWE'})
			datasetHWEDays_1 = xr.merge([datasetFreqDays_1, datasetDuraDays_1, datasetHavgDays_1, datasetTavgDays_1, datasetStartDays_1, datasetEndDays_1])
			datasetHWEDays_1.to_netcdf(outPath_3)

			# datasetTemp_2.utci.plot()
			# plt.show()

# 计算白天-夜间复合高温热浪
def HeatWaveDay(inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, threList_1, outPath_1):
	outPath_2 = os.path.join(outPath_1, '5_HeatWave', 'compound')
	CreateDir(outPath_2)
	for year_1 in range(timeRange_1[0], timeRange_1[1]):
		timeArray_1 = pd.date_range(start = str(year_1) + '-' + timeRange_1[2], end = str(year_1) + '-' + timeRange_1[3])
		outPath_3 = os.path.join(outPath_2, 'hwr_com_' + str(threList_1[0]) + '_' + str(threList_1[1]) + '_' + str(threList_1[2]) + '_' + str(threList_1[3]) + '_' + str(threList_1[4]) + '_' + str(threList_1[5]) + '_' + timeArray_1[0].strftime('%Y%m%d') + '-' + timeArray_1[-1].strftime('%Y%m%d') + '.nc')
		if os.path.exists(outPath_3):
			print(timeArray_1[0].strftime('%Y%m%d') + '-' + timeArray_1[-1].strftime('%Y%m%d') + ' already exists.')
		else:
			datasetTagDay_1 = 0
			datasetTsumDay_1 = 0
			datasetFreqDay_1 = 0 # 频次
			datasetDuraDay_1 = 0 # 持续时间
			datasetTsumDay_2 = 0 # 持续时间

			datasetTagNight_1 = 0
			datasetTsumNight_1 = 0
			datasetFreqNight_1 = 0 # 频次
			datasetDuraNight_1 = 0 # 持续时间
			datasetTsumNight_2 = 0 # 持续时间

			datasetTagDays_1 = 0
			datasetHsumDays_1 = 0
			datasetTsumDays_1 = 0
			datasetFreqDays_1 = 0 # 频次
			datasetDuraDays_1 = 0 # 持续时间
			datasetHsumDays_2 = 0
			datasetTsumDays_2 = 0
			datasetStartDays_1 = 0
			datasetEndDays_1 = 0

			days_2 = timeArray_1[-1] - timeArray_1[0]
			for time_1 in timeArray_1:
				print('compound: ' + time_1.strftime('%Y%m%d'))
				inPath_4 = os.path.join(inPath_1, 'ERA5-HEAT_' + str(year_1), 'ECMWF_utci_' + time_1.strftime('%Y%m%d') + '_v1.1_con.nc')
				inPath_5 = os.path.join(inPath_3, 'day', 'precthre_0.8-0.95_day_2000-' + time_1.strftime('%m-%d') + '.nc')
				inPath_6 = os.path.join(inPath_3, 'night', 'precthre_0.8-0.95_night_2000-' + time_1.strftime('%m-%d') + '.nc')
				try:
					datasetTemp_1 = xr.open_dataset(inPath_4)
					datasetTempThre_1 = xr.open_dataset(inPath_5)
					datasetTempThre_2 = xr.open_dataset(inPath_6)
				except Exception as e:
					print(time_1.strftime('%Y%m%d') + ' does not exists!')		
				datasetTemp_2 = datasetTemp_1.where(datasetMask_1)	# 该小时陆地区域温度
				datasetTempThre_3 = datasetTempThre_1.sel(perc = threList_1[4])
				datasetTempThre_4 = datasetTempThre_2.sel(perc = threList_1[4])
				timeArray_2 = datasetTemp_2['time'].data
				for time_2 in timeArray_2:
					time_2 = pd.to_datetime(time_2)
					datasetTemp_3 = datasetTemp_2.sel(time = time_2)	# 该小时温度

					inPath_7 = os.path.join(inPath_2, 'suntime_2000-' + (time_2 - datetime.timedelta(hours = 1)).strftime('%m-%d_%H') + '.nc')
					inPath_8 = os.path.join(inPath_2, 'suntime_2000-' + time_2.strftime('%m-%d_%H') + '.nc')
					inPath_9 = os.path.join(inPath_2, 'suntime_2000-' + (time_2 + datetime.timedelta(hours = 1)).strftime('%m-%d_%H') + '.nc')
					datasetSuntime_1 = xr.open_dataset(inPath_7)
					datasetSuntime_2 = xr.open_dataset(inPath_8)
					datasetSuntime_3 = xr.open_dataset(inPath_9)

					datasetDayStart_1 = xr.where((1 - datasetSuntime_1) & datasetSuntime_2, 1, 0)	# 将白天/夜间的第一个小时像元标记为1，作为一天的开始
					datasetDayEnd_1 = xr.where(datasetSuntime_2 & (1 - datasetSuntime_3), 1, 0)	# 将白天/夜间的最后一个小时像元标记为1，作为一天的结束
					datasetNightStart_1 = xr.where(datasetSuntime_1 & (1 - datasetSuntime_2), 1, 0)	# 将白天/夜间的第一个小时像元标记为1，作为一天的开始
					datasetNightEnd_1 = xr.where((1 - datasetSuntime_2) & datasetSuntime_3, 1, 0)	# 将白天/夜间的最后一个小时像元标记为1，作为一天的结束

					datasetTemp_4 = xr.where(datasetSuntime_1, datasetTemp_3, np.nan)	# 该小时白天/夜间温度
					datasetTagDay_1, datasetTsumDay_1, datasetFreqDay_1, datasetDuraDay_1, datasetTsumDay_2 = HeatWaveHour(datasetDayStart_1, datasetDayEnd_1, datasetTagDay_1, \
						datasetTsumDay_1, datasetFreqDay_1, datasetDuraDay_1, datasetTsumDay_2, datasetTemp_4, datasetTempThre_3, threList_1[0], threList_1[1])

					datasetTemp_5 = xr.where(1 - datasetSuntime_1, datasetTemp_3, np.nan)	# 该小时白天/夜间温度
					datasetTagNight_1, datasetTsumNight_1, datasetFreqNight_1, datasetDuraNight_1, datasetTsumNight_2 = HeatWaveHour(datasetNightStart_1, datasetNightEnd_1, datasetTagNight_1, \
						datasetTsumNight_1, datasetFreqNight_1, datasetDuraNight_1, datasetTsumNight_2, datasetTemp_5, datasetTempThre_4, threList_1[2], threList_1[3])

					datasetBoolDays_1 = xr.where(datasetNightEnd_1 & (datasetFreqDay_1 >= 1) & (datasetFreqNight_1 >= 1), 1, 0)
					datasetTagDays_1 = xr.where(datasetBoolDays_1, datasetTagDays_1 + 1, datasetTagDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1
					datasetHsumDays_1 = xr.where(datasetBoolDays_1, datasetHsumDays_1 + datasetDuraDay_1 + datasetDuraNight_1, datasetHsumDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1
					datasetTsumDays_1 = xr.where(datasetBoolDays_1, datasetTsumDays_1 + (datasetTsumDay_2 + datasetTsumNight_2) / (datasetDuraDay_1 + datasetDuraNight_1), datasetTsumDays_1)	# 标记为1的像元，即温度高于阈值的像元，累计小时数加1

					datasetCondDays_1 = ((1 - datasetBoolDays_1) & datasetNightEnd_1 | (datasetBoolDays_1 & (time_1.year == timeArray_1[-1].year) & (time_1.month == timeArray_1[-1].month) & (time_1.day == timeArray_1[-1].day))) & (datasetTagDays_1 >= threList_1[5])
					datasetFreqDays_1 = xr.where(datasetCondDays_1, datasetFreqDays_1 + 1, datasetFreqDays_1)	
					datasetDuraDays_1 = xr.where(datasetCondDays_1, datasetDuraDays_1 + datasetTagDays_1, datasetDuraDays_1)
					datasetHsumDays_2 = xr.where(datasetCondDays_1, datasetHsumDays_2 + datasetHsumDays_1, datasetHsumDays_2)
					datasetTsumDays_2 = xr.where(datasetCondDays_1, datasetTsumDays_2 + datasetTsumDays_1, datasetTsumDays_2)
					datasetStartDays_1 = xr.where(datasetCondDays_1 & (datasetStartDays_1 == 0), int(time_1.strftime('%j')) - datasetTagDays_1, datasetStartDays_1)
					datasetEndDays_1 = xr.where(datasetCondDays_1, int(time_1.strftime('%j')), datasetEndDays_1)

					datasetCondDays_2 = ((1 - datasetBoolDays_1) & datasetNightEnd_1) | (time_1.year == timeArray_1[-1].year) & (time_1.month == timeArray_1[-1].month) & (time_1.day == timeArray_1[-1].day)
					datasetHsumDays_1 = xr.where(datasetCondDays_2, 0, datasetHsumDays_1)
					datasetTsumDays_1 = xr.where(datasetCondDays_2, 0, datasetTsumDays_1)
					datasetTagDays_1 = xr.where(datasetCondDays_2, 0, datasetTagDays_1)	# 标记为0的像元，即温度低于阈值的像元，本次高温过程中断，累计小时数初始化为0

			datasetFreqDays_1 = datasetFreqDays_1.where(datasetFreqDays_1 > 0)
			datasetDuraDays_1 = datasetDuraDays_1.where(datasetFreqDays_1 > 0)
			datasetHavgDays_1 = datasetHsumDays_2 / datasetDuraDays_1
			datasetTavgDays_1 = datasetTsumDays_2 / datasetDuraDays_1
			datasetStartDays_1 = datasetStartDays_1.where(datasetFreqDays_1 > 0)
			datasetEndDays_1 = datasetEndDays_1.where(datasetFreqDays_1 > 0)

			datasetFreqDays_1 = datasetFreqDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetDuraDays_1 = datasetDuraDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetHavgDays_1 = datasetHavgDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetTavgDays_1 = datasetTavgDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetStartDays_1 = datasetStartDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))
			datasetEndDays_1 = datasetEndDays_1.where(datasetDuraDays_1 <= (days_2.days + 1))

			datasetFreqDays_1 = datasetFreqDays_1.rename({'utci': 'HWF'})
			datasetDuraDays_1 = datasetDuraDays_1.rename({'utci': 'HWD'})
			datasetHavgDays_1 = datasetHavgDays_1.rename({'utci': 'HWH'})
			datasetTavgDays_1 = datasetTavgDays_1.rename({'utci': 'HWT'})
			datasetStartDays_1 = datasetStartDays_1.rename({'utci': 'HWS'})
			datasetEndDays_1 = datasetEndDays_1.rename({'utci': 'HWE'})
			datasetHWEDays_1 = xr.merge([datasetFreqDays_1, datasetDuraDays_1, datasetHavgDays_1, datasetTavgDays_1, datasetStartDays_1, datasetEndDays_1])
			datasetHWEDays_1.to_netcdf(outPath_3)

			# datasetTemp_3.utci.plot()
			# plt.show()	

def main():
	inPath_1 = r'F:\heatwave\data\1_ERA5-HEAT'
	inPath_2 = r'F:\heatwave\data\2_SunTime'
	inPath_3 = r'F:\heatwave\data\4_PercThre'
	outPath_1 = r'F:\heatwave\data'
	inPath_4 = r'F:\heatwave\data\0_Base\land_extent.nc'

	# # 下载ERA-HEAT数据
	# DownloadData(inPath_1)

	# # 重命名20210101-20210428的数据
	# RenameFile(inPath_1)

	# # 逐小时计算白天/夜间判断矩阵
	# timeArray_1 = pd.date_range(start = '2000-01-01 00:00:00', end = '2000-12-31 23:00:00', freq = 'H')
	# SunTime(timeArray_1, outPath_1)

	# # 计算日白天/夜间平均温度
	# DailyMean(inPath_1, inPath_2, outPath_1)

	# # 计算百分位数阈值（日尺度）
	# timeArray_1 = pd.date_range(start = '2000-01-01', end = '2000-12-31', freq = 'D')
	# PercThre(outPath_1, 'day', timeArray_1, 7, [0.8, 0.85, 0.9, 0.95])
	# PercThre(outPath_1, 'night', timeArray_1, 7, [0.8, 0.85, 0.9, 0.95])
	# # Thread(target = PercThre, args = (outPath_1, 'day', timeArray_1, 7, [0.8, 0.85, 0.9, 0.95])).start()
	# # Thread(target = PercThre, args = (outPath_1, 'night', timeArray_1, 7, [0.8, 0.85, 0.9, 0.95])).start()

	# 计算白天/夜间/复合高温热浪
	timeRange_1 = [1972, 2022, '01-01', '12-31']
	threList_1 = [35, 6, 30, 4, 0.9, 3]
	dayThreList_1 = ['day', threList_1[0], threList_1[4], threList_1[1], threList_1[5]]
	nightThreList_1 = ['night', threList_1[2], threList_1[4], threList_1[3], threList_1[5]]
	datasetMask_1 = xr.open_dataset(inPath_4)
	# HeatWaveSubday(inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, dayThreList_1, outPath_1)
	# HeatWaveSubday(inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, nightThreList_1, outPath_1)
	# HeatWaveDay(inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, threList_1, outPath_1)
	Thread(target = HeatWaveSubday, args = (inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, dayThreList_1, outPath_1)).start()
	Thread(target = HeatWaveSubday, args = (inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, nightThreList_1, outPath_1)).start()
	Thread(target = HeatWaveDay, args = (inPath_1, inPath_2, inPath_3, datasetMask_1, timeRange_1, threList_1, outPath_1)).start()

	# # 测试输出的数据能否正确打开
	# for subday_1 in ['compound', 'day', 'night']:
	# 	outPath_2 = os.path.join(outPath_1, '5_HeatWave', subday_1)
	# 	filenameList_1 = sorted(os.listdir(outPath_2))
	# 	for filename_1 in filenameList_1:
	# 		outPath_3 = os.path.join(outPath_2, filename_1)
	# 		try:
	# 			dataset_1 = xr.open_dataset(outPath_3)
	# 			# dataset_1.HWF.plot()
	# 			# plt.show()	
	# 			print(filename_1 + ' open successfully!')
	# 		except Exception as e:
	# 			print(filename_1)

main()