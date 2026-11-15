# root = 'https://ngdp.sgs.gov.sa/NGDDS/DS_RGP_MAGNETIC/DS_RGP_MAGNETIC_GM_'
# list_codes = ['F','G','H','I','J','K','L','M','N']
# with open("input.txt", 'w') as f:
#     for x in range(0,200):
#         name1 = f'{root}{x:03d}.zip'
#         name2 = f'{root}{x:03d}C.zip'
#         print(name1, file=f)
#         print(name2, file=f)
#     for i in range(16,31):
#         for j in list_codes:
#             name3 = f'{root}{i}{j}.zip'
#             print(name3, file=f)

import pandas as pd

CSV = 'test.csv'
OUT = 'input.txt'
ROOT = 'https://ngdp.sgs.gov.sa/NGDDS/DS_RGP_MAGNETIC/DS_RGP_MAGNETIC_GM_'

# 1:250k sheet -> GM map number (from the reference list of Johnson 2006, SGS-TR-2006-4)
GM = {
    '17F': '077C', '17G': '078C', '18E': '074C', '18F': '075C', '19E': '070C', '19F': '094C',
    '20D': '095C', '20E': '096C', '20F': '097C', '20G': '103C',
    '21D': '107C', '21E': '093C', '21F': '085C', '21G': '090C',
    '22D': '084C', '22E': '088C', '22F': '089C', '22G': '051C', '22H': '092C',
    '23C': '049C', '23D': '087C', '23E': '064C', '23F': '047C', '23G': '046C', '23H': '063C',
    '24B': '062C', '24C': '048C', '24D': '052C', '24E': '058C', '24F': '099C', '24G': '060C',
    '25B': '061C', '25C': '053C', '25E': '028C', '25F': '105C', '25G': '102C',
    '26B': '083C', '26C': '086C', '26D': '106C', '26E': '108C', '26F': '098C',
    '27A': '082C', '27B': '109C', '27E': '115C', '27F': '116C', '28A': '081C', '29A': '080C',
}
ALL = [f'{n:03d}{s}' for n in range(200) for s in ('C', '')]  # fallback for sheets not in GM

df = pd.read_csv(CSV, encoding='utf-8-sig', dtype={'Quad_No': str})
df['sheet'] = df.Min_lat.astype(int).astype(str) + ((df.Min_long - 34.5) // 1.5).astype(int).map(lambda k: chr(65 + k))

with open(OUT, 'w') as f:
    [print(f'{ROOT}{g}_{q}.zip', file=f) for q, s in zip(df.Quad_No, df.sheet) for g in ([GM[s]] if s in GM else ALL)]