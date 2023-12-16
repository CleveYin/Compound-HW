library(tidyverse)
library(ggpubr)
library(readxl)
library(ggpmisc)
library(cartography)
library(rgdal)

# 不同区域热浪
hw_data_1 = read_csv('H:\\1_papers\\3_heat wave\\5_heat wave-carbon emission\\1_data\\4_statistic\\1_mean\\1_zonal\\3_boxplot\\mean_by_region.csv')
hw_data_1$Data <- factor(hw_data_1$Data, levels = c('day', 'night', 'compound'))
hw_data_1$Attribute <- factor(hw_data_1$Attribute, levels = c('HWF', 'HWD', 'HWT', 'HWH', 'HWS', 'HWE'))
hw_data_2 = hw_data_1[hw_data_1$Attribute %in% c('HWF', 'HWD', 'HWT', 'HWH'), ]
ggboxplot(data = hw_data_2, x = 'Region', y = 'Value', fill = 'Data', palette = 'jco', size = 0.4, width = 0.4) + 
  facet_grid(Attribute~Data, scales = "free_y") + ylab(NULL) +  xlab(NULL) + 
  theme_pubr(border = TRUE, legend = 'bottom', x.text.angle = 60) + 
  theme(panel.spacing.y = unit(1, 'cm'), axis.text = element_text(size = 18), 
        legend.title = element_blank(), legend.text = element_text(size = 18), strip.text = element_text(size = 18))

# 热浪逐年变化
hw_data_1 = read_csv('H:\\1_papers\\3_heat wave\\5_heat wave-carbon emission\\1_data\\4_statistic\\1_mean\\hwr_yearly_1972-2021.csv')
hw_data_1$Data <- factor(hw_data_1$Data, levels = c('day', 'night', 'compound'))
hw_data_1$Attribute <- factor(hw_data_1$Attribute, levels = c('HWF', 'HWD', 'HWT', 'HWH', 'HWS', 'HWE'))
hw_data_2 = hw_data_1[hw_data_1$Attribute %in% c('HWF', 'HWD', 'HWH'), ]
Attribute_1 <- c('HWF' = 'HWF (times)', 'HWD' = 'HWD (days)', 'HWH' = 'HWH (hours)')
ggscatter(data = hw_data_2, x = 'Year', y = 'Value', color = 'Data', palette = 'jco', add = 'reg.line', conf.int = TRUE) +
  stat_poly_eq(formula = my.formula, aes(label = paste(after_stat(eq.label), after_stat(p.value.label), sep = "~~~"), color = Data), 
               label.x.npc = 0.05, p.digits = 2, vstep = 0.1, size = 6) +
  facet_wrap(~Attribute, ncol = 3, scales = 'free_y', labeller = as_labeller(Attribute_1)) + ylab(NULL) +  xlab(NULL) +  
  theme_pubr(border = TRUE, legend = 'bottom') + 
  theme(axis.text = element_text(size = 18), strip.text = element_text(size = 18), 
      legend.title = element_blank(), legend.text = element_text(size = 18))

# 热浪逐年变化
phen_grass_1 = read_csv("J:\\其他\\蒙古高原植被物候\\phen_grass.csv")
phen_grass_2 = phen_grass_1[phen_grass_1$Station %in% names(which(table(phen_grass_1$Station) > 2)), ]
ggscatter(data = phen_grass_2, x = 'Year', y = 'SOS', color = 'Station', palette = 'jco', add = 'reg.line', conf.int = TRUE) +
  stat_poly_eq(formula = my.formula, aes(label = paste(after_stat(eq.label), after_stat(p.value.label), sep = "~~~"), color = Station), 
               label.x.npc = 0.05, p.digits = 2, vstep = 0.1, size = 6) +
  facet_wrap('Station', ncol = 4, scales = 'free_y') + ylab(NULL) +  xlab(NULL) +  
  theme_pubr(border = TRUE, legend = 'bottom') + 
  theme(axis.text = element_text(size = 18), strip.text = element_text(size = 18), 
        legend.title = element_blank(), legend.text = element_text(size = 18))

# 热浪逐年变化
hw_data_1 = read_csv("J:\\1_papers\\1_heat wave-carbon emission\\1_data\\4_Statistic\\2_slope\\1_zonal_statistic\\zonal_statistic.csv")
hw_data_1$Data <- factor(hw_data_1$Data, levels = c('day', 'night', 'compound'))
hw_data_1$Attribute <- factor(hw_data_1$Attribute, levels = c('HWF_slope', 'HWD_slope', 'HWH_slope'))
hw_data_2 = hw_data_1[hw_data_1$Attribute %in% c('HWF_slope', 'HWD_slope', 'HWH_slope'), ]
ggbarplot(hw_data_2, x = 'Region', y = 'Value', fill = as.factor(Level), palette = "aaas", size = 0.4, width = 0.4, position = position_fill()) + 
  facet_grid(Attribute~Data, scales = "free_y") + ylab(NULL) +  xlab(NULL) + 
  theme_pubr(border = TRUE, legend = 'bottom', x.text.angle = 60) + 
  theme(text = element_text(size = 18), legend.title = element_blank(), 
        strip.background = element_blank(), strip.text = element_blank())

# figure6: modeled heat waves - historical heat waves
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\2_modeled_summary\\future-historical_HWE_summary.csv")
hw_data_1$Statistic <- factor(hw_data_1$Statistic, levels = c("Recognition rate", "RMSE (HWF)", "RMSE (HWD)"))
ggscatter(hw_data_1, x = "Year", y = "Value", color = "Statistic", palette = "jco", size = 0.5, add = "loess", conf.int = TRUE) +
  facet_wrap(~GCM, ncol = 5) + ylab(NULL) + theme_pubr(x.text.angle = 45, border = TRUE) + theme(legend.title = element_blank())


# figure 9: future heat waves by GCMs
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\2_modeled_summary\\future_yearly_summary.csv")
hw_data_1$Attribute <- factor(hw_data_1$Attribute, levels = c("HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"))
hw_data_2 = hw_data_1[hw_data_1$Attribute %in% c("HWF", "HWD", "HWAT", "HWSD", "HWED"), ]
hw_data_3 = hw_data_2[hw_data_2$GCM %in% c("MRI-ESM2-0", "CNRM-ESM2-1", "CMCC-ESM2", "INM-CM5-0", "EC-EARTH3-VEG-LR", "INM-CM4-8"), ]   # "HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"
ggscatter(hw_data_3, x = "Year", y = "Value", color = "Scenario", palette = "jco", size = 0.5, add = "loess", conf.int = TRUE) +
  facet_grid(Attribute~GCM, scales = "free_y") + ylab(NULL) + theme_pubr(x.text.angle = 45, border = TRUE) + theme(legend.title = element_blank())


# figure 10: exposure density
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\HWD_exposure.csv")
hw_data_1$Society <- factor(hw_data_1$Society, levels = c("Population (billion)", "GDP (trillion)"))
hw_data_1$HWD <- factor(hw_data_1$HWD, levels = c("3-30", "31-60", "61-90", ">91"))
p <- ggplot(hw_data_1, aes(x = Year, y = Exposure, fill = HWD)) + geom_area(color = "black") + 
  facet_grid(Society~Scenario, scales = "free_y") + theme_pubr(border = TRUE)
set_palette(p, "jco")


# figure 11: exposure dotplot
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map\\exposure_by_countries.csv")
hw_data_2 = hw_data_1[hw_data_1$Society %in% c("Population exposure"), ]
ggdotchart(hw_data_2, x = "Country", y = "Exposure", color = "Continent", dot.size = 3, palette = "jco", sorting = "descending", add = "segments") + 
  facet_grid(Society~Year, scales = "free") + xlab(NULL) + ylab("billion people") + 
  theme_pubr(x.text.angle = 45, border = TRUE) + theme(legend.title = element_blank())

hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map\\exposure_by_countries.csv")
hw_data_2 = hw_data_1[hw_data_1$Society %in% c("GDP exposure"), ]
ggdotchart(hw_data_2, x = "Country", y = "Exposure", color = "Continent", dot.size = 3, palette = "jco", sorting = "descending", add = "segments") + 
  facet_grid(Society~Year, scales = "free") + xlab(NULL) + ylab("trillion dollars") + 
  theme_pubr(x.text.angle = 45, border = TRUE) + theme(legend.title = element_blank())

# figure 11: exposure map
my_spdf <- readOGR(dsn = "I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map", layer = "countries_exposure", verbose = FALSE) 
my_spdf@data$pop_1 <- as.numeric(my_spdf@data$pop_1)
my_spdf@data$pop_1[my_spdf@data$pop_1 == 0] <- NA
choroLayer(spdf = my_spdf, df = my_spdf@data, var = "pop_1", col = carto.pal(pal1 = "blue.pal", n1 = 4, pal2 = "red.pal", n2 = 4), colNA = "grey", 
           breaks = c(1e+02, 1e+03, 1e+04, 1e+05, 1e+06, 1e+7, 1e+8, 1e+9, 1e+10), border = "grey", lwd = 0.5, legend.pos = "bottomright", legend.nodata = "no data", legend.horiz = TRUE)

my_spdf <- readOGR(dsn = "I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map", layer = "countries_exposure", verbose = FALSE) 
my_spdf@data$pop_2 <- as.numeric(my_spdf@data$pop_2)
my_spdf@data$pop_2[my_spdf@data$pop_2 == 0] <- NA
choroLayer(spdf = my_spdf, df = my_spdf@data, var = "pop_2", col = carto.pal(pal1 = "blue.pal", n1 = 4, pal2 = "red.pal", n2 = 4), colNA = "grey", 
           breaks = c(1e+02, 1e+03, 1e+04, 1e+05, 1e+06, 1e+7, 1e+8, 1e+9, 1e+10), border = "grey", lwd = 0.5, legend.pos = "bottomright", legend.nodata = "no data", legend.horiz = TRUE)

my_spdf <- readOGR(dsn = "I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map", layer = "countries_exposure", verbose = FALSE) 
my_spdf@data$gdp_1 <- as.numeric(my_spdf@data$gdp_1)
my_spdf@data$gdp_1[my_spdf@data$gdp_1 == 0] <- NA
choroLayer(spdf = my_spdf, df = my_spdf@data, var = "gdp_1", col = carto.pal(pal1 = "green.pal", n1 = 4, pal2 = "purple.pal", n2 = 4), colNA = "grey", 
           breaks = c(1e+06, 1e+07, 1e+08, 1e+09, 1e+10, 1e+11, 1e+12, 1e+13, 1e+14), border = "grey", lwd = 0.5, legend.pos = "bottomright", legend.nodata = "no data", legend.horiz = TRUE)

my_spdf <- readOGR(dsn = "I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\3_exposure\\map", layer = "countries_exposure", verbose = FALSE) 
my_spdf@data$gdp_2 <- as.numeric(my_spdf@data$gdp_2)
my_spdf@data$gdp_2[my_spdf@data$gdp_2 == 0] <- NA
choroLayer(spdf = my_spdf, df = my_spdf@data, var = "gdp_2", col = carto.pal(pal1 = "green.pal", n1 = 4, pal2 = "purple.pal", n2 = 4), colNA = "grey", 
           breaks = c(1e+06, 1e+07, 1e+08, 1e+09, 1e+10, 1e+11, 1e+12, 1e+13, 1e+14), border = "grey", lwd = 0.5, legend.pos = "bottomright", legend.nodata = "no data", legend.horiz = TRUE)
