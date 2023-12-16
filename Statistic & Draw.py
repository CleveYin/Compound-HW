# -*- coding: utf-8 -*-
# @Author: eraer
# @Date:   2022-07-05 16:59:16
# @Last Modified by:   cleve
# @Last Modified time: 2023-11-03 21:03:34

import rioxarray
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt



# import os
# import datetime

# import pygmt

# import numpy as np
# import xarray as xr
# import pandas as pd
# from scipy.stats import linregress
# import matplotlib
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# import cmasher as cmr
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# import matplotlib.ticker as mticker
# import rioxarray

# from scipy import ndimage
# import geopandas as gpd

# import os
# import xarray as xr
# import numpy as np
# import pandas as pd

# from scipy.stats import linregress
# import matplotlib.pyplot as plt
# # import datetime
# # import pygmt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import cartopy.mpl.ticker as cticker
# from cartopy.mpl.geoaxes import GeoAxes
# from cartopy.util import add_cyclic_point
# from cartopy.io.shapereader import Reader
# from cartopy.feature import ShapelyFeature
# import matplotlib
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from mpl_toolkits.axes_grid1 import AxesGrid
# import cmasher as cmr
# # from osgeo import gdal, ogr

# import rioxarray
# import warnings
# warnings.filterwarnings('ignore')

# 白天/夜间判断矩阵示意图
def DrawSunTime(time_1, tag_1, path_1):
	# 生成全球逐小时白天/夜间分区图片，并保存
	fig_1 = pygmt.Figure()
	if tag_1 == 'a':
		fig_1.coast(region = 'd', land = 'darkgreen', water = 'lightblue')
		fig_1.solar(terminator = 'day_night', terminator_datetime = time_1, fill = 'navyblue@50', pen = '0.5p')
		filename_1 = 'fig1-a.pdf'
	elif tag_1 == 'b':
		fig_1.coast(region = 'd', land = 'lightblue', water = 'lightblue')
		fig_1.solar(terminator = 'day_night', terminator_datetime = time_1, fill = 'navyblue@50', pen = '0.5p')
		filename_1 = 'fig1-b.pdf'
	else:
		fig_1.coast(region = 'd', land = 'white', water = 'white')
		fig_1.solar(terminator = 'day_night', terminator_datetime = time_1, fill = 'black', pen = '0.5p')
		filename_1 = 'fig1-c.pdf'
	outPath_1 = os.path.join(path_1, '4_figure', 'source', filename_1)
	fig_1.savefig(outPath_1, dpi = 600)

# Clip the area of interest
def clip(dataset_1, shp_1, field_1, region_1):
	dataset_1.rio.set_spatial_dims("lon", "lat", inplace=True)
	dataset_1.rio.write_crs("epsg:4326", inplace=True)
	if region_1 in shp_1[field_1].tolist():
		shp_2 = shp_1[shp_1[field_1] == region_1]
		dataset_2 = dataset_1.rio.clip(shp_2.geometry.values, shp_2.crs)
	else:
		dataset_2 = dataset_1.rio.clip(shp_1.geometry.values, shp_1.crs)

	return dataset_2

# 计算高温热浪平均值
def CalMean(path_1, data_1, gap_1, yearRange_1):
	inPath_1 = os.path.join(path_1, '1_data', '3_heat wave', data_1)
	outPath_1 = os.path.join(path_1, '1_data', '4_statistic', '1_mean')
	os.makedirs(outPath_1, exist_ok=True)

	filenameDemo_1 = os.listdir(inPath_1)[0].rsplit('_', 1)[0]
	variables_1 = ['HWF', 'HWD', 'HWH', 'HWT', 'HWS', 'HWE']
	lon_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	lat_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)

	datasetList_1 = []
	dataframe_1 = pd.DataFrame(columns = ['Data', 'Year', 'Attribute', 'Value'])
	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
		datasetYear_1 = xr.open_dataset(os.path.join(inPath_1, filenameDemo_1 + '_' + str(year_1) + '0101-' + str(year_1) + '1231.nc'))
		datasetList_1.append(datasetYear_1)

		for variable_1 in variables_1:
			dataframe_1.loc[len(dataframe_1.index)] = [data_1, year_1, variable_1, np.nanmean(datasetYear_1[variable_1].data)]
	datasetYears_1 = xr.concat(datasetList_1, dim = 'time')
	datasetMean_1 = datasetYears_1.mean(dim = 'time', skipna = True)

	# 数据重采样，减少表格数据量或统一空间分辨率
	datasetMean_1 = datasetMean_1.interp(lon = lon_1)
	datasetMean_1 = datasetMean_1.interp(lat = lat_1)
	datasetMean_1 = datasetMean_1.assign_coords(data = data_1)
	datasetMean_1 = datasetMean_1.expand_dims('data')

	# 将数据写入表格
	dataframe_2 = pd.DataFrame(columns = ['Data', 'Attribute', 'Value'])
	for variable_1 in variables_1:
		values_1 = datasetMean_1[variable_1].data.flatten()
		values_2 = values_1[~np.isnan(values_1)].tolist()
		for value_3 in values_2:
			dataframe_2.loc[len(dataframe_2.index)] = [data_1, variable_1, value_3]

	datasetMean_1.to_netcdf(os.path.join(outPath_1, 'hwr_mean_' + data_1 + '_' + str(yearRange_1[0]) + '-' + str(yearRange_1[1]) + '.nc'))		
	print(data_1 + ' is done!')

	return dataframe_1, dataframe_2, datasetMean_1

# 计算单个像元的斜率和P值
def CalSlopePixel(y_1):
	x_1 = np.arange(len(y_1))	# 自变量
	nanCount_1 = np.count_nonzero(np.isnan(y_1))	# 计算因变量中空值的数量
	if nanCount_1 >= 0 and nanCount_1 <= len(y_1) * 2 / 5:	# 如果因变量中空值的数量少于三分之一，则删除空值计算斜率
		nanPos_1 = np.argwhere(np.isnan(y_1))	# 因变量中空值的索引
		nanPos_2 = nanPos_1.flatten().tolist()
		x_1 = np.delete(x_1, nanPos_2)	# 删除空值位置上对应的自变量
		y_1 = y_1[~np.isnan(y_1)]	# 删除因变量中的空值
	result = linregress(x_1, y_1)
	slope = result.slope
	pvalue = result.pvalue
	pvalue_1 = np.nan
	pvalue_2 = np.nan
	if pvalue > 0.05:
		if slope < 0:
			slope = np.nan
			pvalue_1 = -1
		elif slope >= 0:
			slope = np.nan
			pvalue_2 = 1
	return slope, pvalue_1, pvalue_2

# 计算多年高温热浪的斜率和P值
def CalSlope(path_1, data_1, gap_1, yearRange_1):
	inPath_1 = os.path.join(path_1, '1_data', '3_heat wave', data_1)
	outPath_1 = os.path.join(path_1, '1_data', '4_statistic', '2_slope')
	os.makedirs(outPath_1, exist_ok=True)

	filenameDemo_1 = os.listdir(inPath_1)[0].rsplit('_', 1)[0]
	variables_1 = ['HWF', 'HWD', 'HWH', 'HWT', 'HWS', 'HWE']
	lon_1 = np.arange(-180 + gap_1 / 2, 180 + gap_1 / 2, gap_1)
	lat_1 = np.arange(-90 + gap_1 / 2, 90 + gap_1 / 2, gap_1)

	datasetList_1 = []
	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
		datasetYear_1 = xr.open_dataset(os.path.join(inPath_1, filenameDemo_1 + '_' + str(year_1) + '0101-' + str(year_1) + '1231.nc'))
		datasetList_1.append(datasetYear_1)
	datasetYears_1 = xr.concat(datasetList_1, dim = 'time')

	datasetYears_1 = datasetYears_1.interp(lon = lon_1)
	datasetYears_1 = datasetYears_1.interp(lat = lat_1)

	datasetList_2 = []
	for variable_1 in variables_1:
		results_1 = np.apply_along_axis(CalSlopePixel, 0, datasetYears_1[variable_1].data)
		# plt.imshow(results_1[0])
		# plt.show()
		datasetResult_1 = xr.Dataset({variable_1 + '_slope': (['lat', 'lon'], results_1[0]), 
									variable_1 + '_pvalue_1': (['lat', 'lon'], results_1[1]), 
									variable_1 + '_pvalue_2': (['lat', 'lon'], results_1[2])}, 
									coords = {'lat': lat_1, 'lon': lon_1})
		datasetList_2.append(datasetResult_1)
		print(variable_1 + ' of ' + data_1 + ' is done!')
	datasetHWE_1 = xr.merge(datasetList_2)
	datasetHWE_1 = datasetHWE_1.assign_coords(data = data_1)
	datasetHWE_1 = datasetHWE_1.expand_dims('data')

	datasetHWE_1.to_netcdf(os.path.join(outPath_1, 'hwr_slope_' + data_1 + '_' + str(yearRange_1[0]) + '-' + str(yearRange_1[1]) + '.nc'))
	return datasetHWE_1

# 绘制热浪平均值（三个子图）
def DrawRaster(tag_1, variable_1, datasetList_1, cmap_1, title_1, path_1):
	# 读取数据
	variables_1 = list(datasetList_1[0].keys())

	# 创建分级
	dataArrayList_1 = []
	for variable_2 in variables_1:
		dataArray_1 = datasetList_1[0][variable_2]
		dataArrayList_1.append(dataArray_1)
	dataArray_2 = np.dstack(dataArrayList_1)
	if tag_1 == 'mean':
		figname_1 = 'fig2_'
		interval_1 = int((np.nanmax(dataArray_2) - np.nanmin(dataArray_2)) / 10)
		levels_1 = np.arange(int(np.nanmin(dataArray_2)), int(np.nanmax(dataArray_2)), interval_1)
		if (np.nanmax(dataArray_2) - levels_1[-1]) >= (interval_1 / 3):
			levels_1 = np.append(levels_1, levels_1[-1] + interval_1)
	elif tag_1 == 'slope':
		figname_1 = 'fig4_'
		absMin_1 = round(abs(np.nanmin(dataArray_2)), 2)
		absMax_1 = round(abs(np.nanmax(dataArray_2)), 2)
		absEdge_1 = absMin_1 if absMin_1 > absMax_1 else absMax_1
		interval_1 = round(2 * absEdge_1 / 10, 2)
		levelList_1 = [0]
		for index_1 in range(5):
			levelList_1.append(round(interval_1 * (index_1 + 1), 2))
			levelList_1.append(round(- interval_1 * (index_1 + 1), 2))
		levels_1 = sorted(levelList_1)
	print(np.nanmin(dataArray_2), np.nanmax(dataArray_2))
	print(levels_1)

	# 自定义颜色系
	cmap_2 = matplotlib.cm.get_cmap(cmap_1, lut = len(levels_1) - 1)
	if cmap_1 in ['tab20c', 'tab20c_r']:
		cmap_2_1 = cmr.get_sub_cmap(cmap_1, 0.8, 1)
		cmap_2_2 = cmr.get_sub_cmap(cmap_1, 0.4, 0.6)
		cmap_2_3 = cmr.get_sub_cmap(cmap_1, 0.6, 0.8)
		cmap_2_4 = cmr.get_sub_cmap(cmap_1, 0.2, 0.4)
		quotient_1 = int((len(levels_1) - 1) / 4)
		remainder_1 = int((len(levels_1) - 1) % 4)
		countArray_1 = np.full(4, quotient_1)
		if remainder_1 == 1:
			countArray_1[3] = quotient_1 + 1
		elif remainder_1 == 2:
			countArray_1[2] = quotient_1 + 1
			countArray_1[3] = quotient_1 + 1
		elif remainder_1 == 3:
			countArray_1[1] = quotient_1 + 1
			countArray_1[2] = quotient_1 + 1
			countArray_1[3] = quotient_1 + 1
		cmap_2_5 = np.vstack((cmap_2_1(np.linspace(0, 1, countArray_1[0])), cmap_2_2(np.linspace(0, 1, countArray_1[1])), cmap_2_3(np.linspace(0, 1, countArray_1[2])), cmap_2_4(np.linspace(0, 1, countArray_1[3]))))
		cmap_2_6 = ListedColormap(cmap_2_5, name = 'i' + cmap_1)
		cmap_2 = matplotlib.cm.get_cmap(cmap_2_6, lut = len(levels_1) - 1)
	colorList_1 = []
	for index_1 in range(cmap_2.N):
		colorList_1.append(matplotlib.colors.rgb2hex(cmap_2(index_1)))

	fig_1, axs_1 = plt.subplots(nrows = 1, ncols = 3, subplot_kw = {'projection': ccrs.PlateCarree()}, figsize = (15, 2.5))
	axs_1 = axs_1.flatten()
	for index_1, variable_3 in enumerate(variables_1):
		dataArray_3 = datasetList_1[0][variable_3]

		cs_1 = axs_1[index_1].pcolor(datasetList_1[0]['lon'], datasetList_1[0]['lat'], dataArray_3, cmap = cmap_2, vmin = levels_1[0], vmax = levels_1[-1])	# 绘制风险使用
		axs_1[index_1].set_extent([-179.99, 180, -58, 85])

		# # 绘制地名标签
		# if index_1 == 1 and (variable_1 == 'HWH'):
		# 	axs_1[index_1].add_feature(ShapelyFeature(Reader(r'F:\heatwave\data\0_Base\World_Regions_Con_Abb.shp').geometries(), ccrs.PlateCarree()), linewidth = 0.5, facecolor = 'none')
		# 	# axs_1[index_1].gridlines(draw_labels = True)
		# 	dataframe_1 = pd.read_csv(r'F:\heatwave\data\0_Base\region_name_loc.csv')
		# 	lon_1 = dataframe_1['lon'].to_list()
		# 	lat_1 = dataframe_1['lat'].to_list()
		# 	name_1 = (dataframe_1['REGION_ABB']).to_list()
		# 	nameZip_1 = zip(lon_1, lat_1, name_1)
		# 	for (lon_2, lat_2, name_2) in nameZip_1:
		# 		axs_1[index_1].text(lon_2, lat_2, name_2, fontsize = 18, color = 'm')

		axs_1[index_1].coastlines(zorder = 4)

		if tag_1 == 'slope':
			gridlines_1 = axs_1[index_1].gridlines(crs = ccrs.PlateCarree(), linewidth = 0.1, color = 'black', alpha = 1, zorder = 1)
			gridlines_1.xlocator = mticker.LinearLocator(180)
			gridlines_1.ylocator = mticker.LinearLocator(90)

			dataArray_4 = np.where(np.isnan(dataArray_3.data), 1, np.nan)
			axs_1[index_1].contourf(datasetList_1[1]['lon'], datasetList_1[0]['lat'], dataArray_4, colors = "white", zorder = 2)

			dataArray_5 = datasetList_1[1][variable_3]
			dataArray_6 = datasetList_1[2][variable_3]
			axs_1[index_1].contourf(datasetList_1[1]['lon'], datasetList_1[0]['lat'], dataArray_5, colors = "yellow", zorder = 3)
			axs_1[index_1].contourf(datasetList_1[2]['lon'], datasetList_1[0]['lat'], dataArray_6, colors = "green", zorder = 3)

		countList_1 = []
		for index_2 in range(1, len(levels_1)):
			dataArray_7 = np.where((dataArray_3 >= levels_1[index_2 - 1]) & (dataArray_3 < levels_1[index_2]), 1, 0)
			count_1 = np.count_nonzero(dataArray_7)
			countList_1.append(count_1)
		ax_2 = fig_1.add_axes([-0.15 + index_1 * 0.33, 0.18, 0.4, 0.4])
		ax_2.pie(countList_1, radius = 0.8, wedgeprops = {'width': 0.3, 'edgecolor':'k','linewidth': 0.6}, colors = colorList_1)

	fig_1.subplots_adjust(bottom = 0.12, top = 1, left = 0.01, right = 0.99, wspace = 0.02, hspace = 0.02)
	cax_1 = fig_1.add_axes([0.15, 0.12, 0.7, 0.04])
	cbar_1 = fig_1.colorbar(cs_1, cax = cax_1, orientation = 'horizontal', extend = 'both', ticks = levels_1)
	cbar_1.set_ticklabels(levels_1)	
	cbar_1.ax.tick_params(labelsize = 18)

	# plt.suptitle(title_1, x = 0.01, ha = 'left', fontsize = 18)
	fig_1.text(0.88, 0.02, title_1, fontsize = 18)
	outPath_1 = os.path.join(path_1, '4_figure', 'source', figname_1 + variable_1 + '.pdf')
	plt.savefig(outPath_1)
	# plt.show()

# 将NC数据（分级）并另存为栅格数据
def NcToTiff(tag_1, dataset_1, classRefList_1, path_1):
	os.makedirs(path_1, exist_ok=True)
	if tag_1 == 'mean':
		attriList_1 = ['HWF', 'HWD', 'HWT', 'HWH']
	elif tag_1 == 'slope':
		attriList_1 = ['HWF_slope', 'HWD_slope', 'HWT_slope', 'HWH_slope']
	index_1 = 0
	for attri_1 in attriList_1:
		index_2 = 0
		for data_1 in ['compound', 'day', 'night']:
			dataArray_1 = dataset_1[attri_1][index_2]
			# dataArray_1 = xr.where((dataArray_1 >= classRefList_1[index_1][0]) & (dataArray_1 < classRefList_1[index_1][1]), 1, dataArray_1)
			# dataArray_1 = xr.where((dataArray_1 >= classRefList_1[index_1][1]) & (dataArray_1 < classRefList_1[index_1][2]), 2, dataArray_1)
			# dataArray_1 = xr.where((dataArray_1 >= classRefList_1[index_1][2]) & (dataArray_1 < classRefList_1[index_1][3]), 3, dataArray_1)
			# dataArray_1 = xr.where((dataArray_1 >= classRefList_1[index_1][3]), 4, dataArray_1)
			# dataArray_1.rio.write_nodata(-9999, inplace = True)
			dataArray_1.rio.set_spatial_dims("lon", "lat", inplace = True)
			dataArray_1.rio.write_crs("epsg:4326", inplace = True)
			outPath_1 = os.path.join(path_1, 'hwr_' + tag_1 + '_' + attri_1 + '_' + data_1 + '_1972-2021.tif')
			dataArray_1.rio.to_raster(outPath_1)
			index_2 = index_2 + 1
		index_1 = index_1 + 1

# 重构表格数据，以满足制图需要
def MergeTable(path_1):
	inPath_1 = os.path.join(path_1, '1_data', '4_statistic', '1_mean', 'map', '2_xls')
	for attri_1 in ['HWF', 'HWD', 'HWT', 'HWH']:
		dataframes_1 = []
		for data_1 in ['day', 'night', 'compound']:
			dataframe_1 = pd.read_excel(os.path.join(inPath_1, 'hwr_mean_' + attri_1 + '_' + data_1 + '_1972-2021.tif.xls'))
			dataframe_1['data'] = [data_1 for index_1 in range(len(dataframe_1))]
			dataframes_1.append(dataframe_1)
		dataframe_2 = pd.concat(dataframes_1)

		locations_1 = set(dataframe_2['REGION_ABB'])
		index_2 = 0
		dataframe_3 = pd.DataFrame(columns = ['REGION_ABB', 'day', 'night', 'compound'])
		for location_1 in locations_1:
			record_1 = [location_1]
			for data_2 in ['day', 'night', 'compound']:
				dataframe_4 = dataframe_2[(dataframe_2['REGION_ABB'] == location_1) & (dataframe_2['data'] == data_2)]
				dataframe_4.reset_index(inplace = True)
				try:
					record_1.append(dataframe_4.loc[0, 'MEAN'])
				except Exception as e:
					record_1.append(np.nan)
			dataframe_3.loc[len(dataframe_3.index)] = record_1
		dataframe_3.to_csv(os.path.join(path_1, '1_data', '4_statistic', '1_mean', 'map', '3_merge', attri_1 + '.csv'), index = False)
		print(attri_1)

# 合并表格数据
def ComTableMean(path_1):
	inPath_1 = os.path.join(path_1, '1_data', '4_statistic', '1_mean', '1_zonal', '3_boxplot', '1_xls')
	outPath_1 = os.path.join(path_1, '1_data', '4_statistic', '1_mean', '1_zonal', '3_boxplot', 'mean_by_region.csv')
	dataframe_1 = pd.DataFrame(columns = ['Attribute', 'Data', 'Region', 'Value'])
	attriList_1 = []
	dataList_1 = []
	regionList_1 = []
	valueList_1 = []

	for attri_1 in ['HWF', 'HWD', 'HWT', 'HWH']:
		for data_1 in ['compound', 'day', 'night']:
			inPath_2 = os.path.join(inPath_1, 'hwr_mean_' + attri_1 + '_' + data_1 + '_1972-2021.tif.xls')
			dataframe_2 = pd.read_excel(inPath_2)
			dataframe_2 = dataframe_2[['REGION_ABB', 'MIN', 'MEAN', 'MAX']]
			regionList_2 = dataframe_2['REGION_ABB'].tolist()
			for region_1 in regionList_2:
				dataframe_3 = dataframe_2[dataframe_2['REGION_ABB'] == region_1]
				valueList_2 = dataframe_3.iloc[0].tolist()
				attriList_1 = attriList_1 + [attri_1, attri_1, attri_1]
				dataList_1 = dataList_1 + [data_1, data_1, data_1]
				regionList_1 = regionList_1 + [valueList_2[0], valueList_2[0], valueList_2[0]]
				valueList_1 = valueList_1 + valueList_2[1: ]
	dataframe_1['Attribute'] = attriList_1
	dataframe_1['Data'] = dataList_1
	dataframe_1['Region'] = regionList_1
	dataframe_1['Value'] = valueList_1	
	dataframe_1.to_csv(outPath_1, index = False)

# 合并表格数据
def ComTableSlope(inPath_1, outPath_1):
	dataframe_1 = pd.DataFrame(columns = ['Attribute', 'Data', 'Region', 'Level', 'Value'])
	attriList_1 = []
	dataList_1 = []
	regionList_1 = []
	levelList_1 = []
	valueList_1 = []

	for attri_1 in ['HWF_slope', 'HWD_slope', 'HWT_slope', 'HWH_slope']:
		for data_1 in ['compound', 'day', 'night']:
			inPath_2 = os.path.join(inPath_1, 'hwr_mean_' + attri_1 + '_' + data_1 + '_1979-2021.tif.xls')
			dataframe_2 = pd.read_excel(inPath_2)
			levelList_2 = dataframe_2['LABEL'].tolist()
			for levels_1 in levelList_2:
				dataframe_3 = dataframe_2[dataframe_2['LABEL'] == levels_1]
				regionList_2 = dataframe_3.columns.tolist()
				valueList_2 = dataframe_3.iloc[0].tolist()
				# print(regionList_2[2: ])
				# print(valueList_2[2: ])
				for region_1 in regionList_2[2: ]:
					attriList_1.append(attri_1)
					dataList_1.append(data_1)
					levelList_1.append(levels_1)
				regionList_1 = regionList_1 + regionList_2[2: ]
				valueList_1 = valueList_1 + valueList_2[2: ]
	dataframe_1['Attribute'] = attriList_1
	dataframe_1['Data'] = dataList_1
	dataframe_1['Region'] = regionList_1
	dataframe_1['Level'] = levelList_1
	dataframe_1['Value'] = valueList_1	
	dataframe_1.to_csv(outPath_1, index = False)

# 计算日间、夜间、复合热浪的比例变化
def CompareHW(path_1, datas_1, yearRange_1):
	inPath_1 = os.path.join(path_1, '1_data', '3_heat wave')
	outPath_1 = os.path.join(path_1, '1_data', '4_statistic', '3_compare')
	os.makedirs(outPath_1, exist_ok=True)

	variables_1 = ['HWF', 'HWD', 'HWH', 'HWT', 'HWS', 'HWE']

	for year_1 in range(yearRange_1[0], yearRange_1[1] + 1):
		dataset_1 = xr.open_dataset(os.path.join(inPath_1, datas_1[0], 'hwr_' + datas_1[0] + '_35_0.9_6_3_' + str(year_1) +'0101-' + str(year_1) + '1231.nc'))
		dataset_2 = xr.open_dataset(os.path.join(inPath_1, datas_1[1], 'hwr_' + datas_1[1] + '_30_0.9_4_3_' + str(year_1) +'0101-' + str(year_1) + '1231.nc'))

		dataset_3 = dataset_2 / dataset_1
		outPath_2 = os.path.join(outPath_1, 'hwr_compare_' + str(year_1) +'0101-' + str(year_1) + '1231.nc')
		dataset_3.to_netcdf(outPath_2)
		print(year_1)

def main():
	path_1 = r'H:\1_papers\3_heat wave\5_heat wave-carbon emission'

	# # 白天/夜间判断矩阵示意图
	# time_1 = datetime.datetime(year = 2020, month = 7, day = 1, hour = 12, minute = 0, second = 0)
	# DrawSunTime(time_1, 'c', path_1)

	dataset_1 = xr.open_dataset(r"F:\1_papers\3_heat wave\5_heat wave-carbon emission\1_data\3_heat wave\day\hwr_day_35_0.9_6_3_20210101-20211231.nc")
	shp_1 = gpd.read_file(r"F:\1_papers\3_heat wave\5_heat wave-carbon emission\3_code\CSJ_shp\CSJ_cities_polygon_EN_name.shp")
	dataset_2 = clip(dataset_1, shp_1, 'NAME_2', 'CSJ')
	# print(dataset_2)

	array_1 = dataset_2['HWD'].values
	plt.imshow(array_1)
	plt.show()

	# # 计算历史高温热浪多年平均值和变化率
	# dataframeList_1 = []
	# dataframeList_2 = []
	# datasetList_1 = []
	# datasetList_2 = []
	# dataList_1 = ['day', 'night', 'compound']
	# for data_1 in dataList_1:
	# 	dataframe_1, dataframe_2, dataset_1 = CalMean(path_1, data_1, 1, [1972, 2021])
	# 	dataset_2 = CalSlope(path_1, data_1, 1, [1972, 2021])
	# 	dataframeList_1.append(dataframe_1)
	# 	dataframeList_2.append(dataframe_2)
	# 	datasetList_1.append(dataset_1)
	# 	datasetList_2.append(dataset_2)
	# dataframe_3 = pd.concat(dataframeList_1)
	# dataframe_4 = pd.concat(dataframeList_2)
	# dataset_3 = xr.merge(datasetList_1)
	# dataset_4 = xr.merge(datasetList_2)
	# dataframe_3.to_csv(os.path.join(path_1, '1_data', '4_statistic', '1_mean', 'hwr_yearly_1972-2021.csv'), index = False)
	# dataframe_4.to_csv(os.path.join(path_1, '1_data', '4_statistic', '1_mean', 'hwr_mean_1972-2021.csv'), index = False)
	# dataset_3.to_netcdf(os.path.join(path_1, '1_data', '4_statistic', '1_mean', 'hwr_mean_1972-2021.nc'))
	# dataset_4.to_netcdf(os.path.join(path_1, '1_data', '4_statistic', '2_slope', 'hwr_slope_1972-2021.nc'))

	# # 绘制平均值
	# dataset_1 = xr.open_dataset(r"H:\1_papers\3_heat wave\5_heat wave-carbon emission\1_data\4_statistic\1_mean\hwr_mean_1972-2021.nc")
	# cmapList_1 = ['plasma_r', 'plasma_r', 'tab20c_r', 'plasma_r', 'viridis_r', 'viridis_r']
	# titleList_1 = ['times', 'days', 'hours', '°C', 'DOY', 'DOY']
	# index_1 = 0
	# for variable_1 in ['HWF', 'HWD', 'HWH', 'HWT', 'HWS', 'HWE']:
	# 	dataset_2 = xr.Dataset({'day': (['lat', 'lon'], dataset_1[variable_1].data[1]), 
	# 							'night': (['lat', 'lon'], dataset_1[variable_1].data[2]), 
	# 							'compound': (['lat', 'lon'], dataset_1[variable_1].data[0])}, 
	# 							coords = {'lon': dataset_1['lon'].data, 'lat': dataset_1['lat'].data})
	# 	if variable_1 == 'HWT':
	# 		dataset_2 = dataset_2 - 273.15
	# 	DrawRaster('mean', variable_1, [dataset_2], cmapList_1[index_1], titleList_1[index_1], path_1)
	# 	index_1 = index_1 + 1

	# # 绘制斜率
	# dataset_1 = xr.open_dataset(r"H:\1_papers\3_heat wave\5_heat wave-carbon emission\1_data\4_statistic\2_slope\hwr_slope_1972-2021.nc")
	# titleList_1 = ['times / year', 'days / year', 'hours / year', '°C / year', 'DOY / year', 'DOY / year']
	# index_1 = 0
	# for variable_1 in ['HWF', 'HWD', 'HWH', 'HWT', 'HWS', 'HWE']:
	# 	variable_2 = variable_1 + '_slope'
	# 	variable_3 = variable_1 + '_pvalue_1'
	# 	variable_4 = variable_1 + '_pvalue_2'
	# 	dataset_2 = xr.Dataset({'day': (['lat', 'lon'], dataset_1[variable_2].data[1]), 
	# 							'night': (['lat', 'lon'], dataset_1[variable_2].data[2]), 
	# 							'compound': (['lat', 'lon'], dataset_1[variable_2].data[0])}, 
	# 							coords = {'lon': dataset_1['lon'].data, 'lat': dataset_1['lat'].data})
	# 	dataset_3 = xr.Dataset({'day': (['lat', 'lon'], dataset_1[variable_3].data[1]), 
	# 							'night': (['lat', 'lon'], dataset_1[variable_3].data[2]), 
	# 							'compound': (['lat', 'lon'], dataset_1[variable_3].data[0])}, 
	# 							coords = {'lon': dataset_1['lon'].data, 'lat': dataset_1['lat'].data})
	# 	dataset_4 = xr.Dataset({'day': (['lat', 'lon'], dataset_1[variable_4].data[1]), 
	# 							'night': (['lat', 'lon'], dataset_1[variable_4].data[2]), 
	# 							'compound': (['lat', 'lon'], dataset_1[variable_4].data[0])}, 
	# 							coords = {'lon': dataset_1['lon'].data, 'lat': dataset_1['lat'].data})
	# 	DrawRaster('slope', variable_1, [dataset_2, dataset_3, dataset_4], 'RdBu_r', titleList_1[index_1], path_1)
	# 	index_1 = index_1 + 1
 
	# # 将NC数据（分级）并另存为栅格数据
	# index_1 = 1
	# for tag_1 in ['mean']:
	# 	dataset_1 = xr.open_dataset(os.path.join(path_1, '1_data', '4_statistic', str(index_1) + '_' + tag_1, 'hwr_' + tag_1 + '_1972-2021.nc'))
	# 	NcToTiff(tag_1, dataset_1, [], os.path.join(path_1, '1_data', '4_statistic', str(index_1) + '_' + tag_1, '1_zonal', '1_tif'))
	# 	index_1 = index_1 + 1

	# # 重构表格数据，以满足制图需要
	# MergeTable(path_1)

	# CompareHW(path_1, ['day', 'night'], [1972, 2021])

	# # 合并表格数据
	# ComTableMean(path_1)

	# # # 合并表格数据
	# inPath_2 = r'J:\1_papers\1_heat wave-carbon emission\1_data\4_Statistic\1_mean\1_mean_by_region\2_table'
	# outPath_3 = r'J:\1_papers\1_heat wave-carbon emission\1_data\4_Statistic\1_mean\1_mean_by_region\mean_by_region.csv' 
	# ComTable(inPath_2, outPath_3)

	# # 合并表格数据
	# inPath_2 = r'J:\1_papers\1_heat wave-carbon emission\1_data\4_Statistic\2_slope\1_zonal_statistic\2_table'
	# outPath_3 = r'J:\1_papers\1_heat wave-carbon emission\1_data\4_Statistic\2_slope\1_zonal_statistic\zonal_statistic.csv' 
	# ComTableSlope(inPath_2, outPath_3)
	
main()
