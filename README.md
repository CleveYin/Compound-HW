# Development of "Real" Compound Heat Wave Datasets and Their Impacts on Carbon Emissions

Nighttime HWs may pose a greater threat to human health because they may lead to a failure to naturally recover from daytime HWs and may even exacerbate injuries due to reduced nighttime mobility and medical availability.

Based on using hourly climate data, we have added a duration hour threshold to avoid identifying short-term high temperature processes as HW days. Only weather processes that meet the high temperature threshold, duration hour threshold and duration day threshold can be identified as a HW event. 

We calculated the "real" daytime, nighttime, and compound HWs. Specifically, for each pixel, the daily sunrise and sunset times are firstly calculated according to the local latitude, longitude, and date. The daytime (nighttime) HW is strictly limited between sunrise (sunset) and the next sunset (sunrise), and the compound HW is strictly limited between two sunrises.
