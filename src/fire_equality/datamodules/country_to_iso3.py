"""
国家名称到ISO3代码的映射
用于将FireTracks数据中的country列转换为iso3代码
"""
import pandas as pd

# 国家名称到ISO3代码的映射字典
# 这个映射基于常见的国家名称变体
COUNTRY_TO_ISO3 = {
    # 主要国家（常见变体）
    'United States': 'USA',
    'United States of America': 'USA',
    'USA': 'USA',
    'US': 'USA',
    'U.S.A.': 'USA',
    'United States of America (the)': 'USA',
    
    'Australia': 'AUS',
    'AUS': 'AUS',
    
    'Canada': 'CAN',
    'CAN': 'CAN',
    
    'Brazil': 'BRA',
    'Brasil': 'BRA',
    'BRA': 'BRA',
    
    'China': 'CHN',
    'People\'s Republic of China': 'CHN',
    'CHN': 'CHN',
    
    'Russia': 'RUS',
    'Russian Federation': 'RUS',
    'RUS': 'RUS',
    
    'India': 'IND',
    'IND': 'IND',
    
    'Argentina': 'ARG',
    'ARG': 'ARG',
    
    'Mexico': 'MEX',
    'MEX': 'MEX',
    
    'South Africa': 'ZAF',
    'ZAF': 'ZAF',
    
    'Indonesia': 'IDN',
    'IDN': 'IDN',
    
    'Chile': 'CHL',
    'CHL': 'CHL',
    
    'Peru': 'PER',
    'PER': 'PER',
    
    'Colombia': 'COL',
    'COL': 'COL',
    
    'Venezuela': 'VEN',
    'VEN': 'VEN',
    
    'Bolivia': 'BOL',
    'BOL': 'BOL',
    
    'Paraguay': 'PRY',
    'PRY': 'PRY',
    
    'Uruguay': 'URY',
    'URY': 'URY',
    
    'Ecuador': 'ECU',
    'ECU': 'ECU',
    
    'Guyana': 'GUY',
    'GUY': 'GUY',
    
    'Suriname': 'SUR',
    'SUR': 'SUR',
    
    'French Guiana': 'GUF',
    'GUF': 'GUF',
    
    # 非洲国家
    'Central African Republic': 'CAF',
    'CAF': 'CAF',
    
    'Nigeria': 'NGA',
    'NGA': 'NGA',
    
    'Democratic Republic of the Congo': 'COD',
    'DRC': 'COD',
    'Congo (Democratic Republic of the)': 'COD',
    'COD': 'COD',
    
    'South Sudan': 'SSD',
    'SSD': 'SSD',
    
    'Chad': 'TCD',
    'TCD': 'TCD',
    
    'Ethiopia': 'ETH',
    'ETH': 'ETH',
    
    'Sudan': 'SDN',
    'SDN': 'SDN',
    
    'Mozambique': 'MOZ',
    'MOZ': 'MOZ',
    
    'Angola': 'AGO',
    'AGO': 'AGO',
    
    'Zambia': 'ZMB',
    'ZMB': 'ZMB',
    
    'Zimbabwe': 'ZWE',
    'ZWE': 'ZWE',
    
    'Botswana': 'BWA',
    'BWA': 'BWA',
    
    'Namibia': 'NAM',
    'NAM': 'NAM',
    
    'Tanzania': 'TZA',
    'United Republic of Tanzania': 'TZA',
    'TZA': 'TZA',
    
    'Kenya': 'KEN',
    'KEN': 'KEN',
    
    'Uganda': 'UGA',
    'UGA': 'UGA',
    
    'Ghana': 'GHA',
    'GHA': 'GHA',
    
    'Ivory Coast': 'CIV',
    "Côte d'Ivoire": 'CIV',
    'CIV': 'CIV',
    
    'Guinea': 'GIN',
    'GIN': 'GIN',
    
    'Mali': 'MLI',
    'MLI': 'MLI',
    
    'Niger': 'NER',
    'NER': 'NER',
    
    'Burkina Faso': 'BFA',
    'BFA': 'BFA',
    
    'Senegal': 'SEN',
    'SEN': 'SEN',
    
    'Cameroon': 'CMR',
    'CMR': 'CMR',
    
    'Gabon': 'GAB',
    'GAB': 'GAB',
    
    'Republic of the Congo': 'COG',
    'Congo': 'COG',
    'COG': 'COG',
    
    'Benin': 'BEN',
    'BEN': 'BEN',
    
    'Togo': 'TGO',
    'TGO': 'TGO',
    
    'Liberia': 'LBR',
    'LBR': 'LBR',
    
    'Sierra Leone': 'SLE',
    'SLE': 'SLE',
    
    'Gambia': 'GMB',
    'GMB': 'GMB',
    
    'Guinea-Bissau': 'GNB',
    'GNB': 'GNB',
    
    'Mauritania': 'MRT',
    'MRT': 'MRT',
    
    'Malawi': 'MWI',
    'MWI': 'MWI',
    
    'Madagascar': 'MDG',
    'MDG': 'MDG',
    
    'Rwanda': 'RWA',
    'RWA': 'RWA',
    
    'Burundi': 'BDI',
    'BDI': 'BDI',
    
    'Eritrea': 'ERI',
    'ERI': 'ERI',
    
    'Djibouti': 'DJI',
    'DJI': 'DJI',
    
    'Somalia': 'SOM',
    'SOM': 'SOM',
    
    # 亚洲国家
    'Kazakhstan': 'KAZ',
    'KAZ': 'KAZ',
    
    'Mongolia': 'MNG',
    'MNG': 'MNG',
    
    'Myanmar': 'MMR',
    'Burma': 'MMR',
    'MMR': 'MMR',
    
    'Thailand': 'THA',
    'THA': 'THA',
    
    'Vietnam': 'VNM',
    'Viet Nam': 'VNM',
    'VNM': 'VNM',
    
    'Cambodia': 'KHM',
    'KHM': 'KHM',
    
    'Laos': 'LAO',
    "Lao People's Democratic Republic": 'LAO',
    'LAO': 'LAO',
    
    'Malaysia': 'MYS',
    'MYS': 'MYS',
    
    'Philippines': 'PHL',
    'PHL': 'PHL',
    
    'Papua New Guinea': 'PNG',
    'PNG': 'PNG',
    
    'Bangladesh': 'BGD',
    'BGD': 'BGD',
    
    'Pakistan': 'PAK',
    'PAK': 'PAK',
    
    'Afghanistan': 'AFG',
    'AFG': 'AFG',
    
    'Iran': 'IRN',
    'Islamic Republic of Iran': 'IRN',
    'IRN': 'IRN',
    
    'Iraq': 'IRQ',
    'IRQ': 'IRQ',
    
    'Saudi Arabia': 'SAU',
    'SAU': 'SAU',
    
    'Yemen': 'YEM',
    'YEM': 'YEM',
    
    'Oman': 'OMN',
    'OMN': 'OMN',
    
    'United Arab Emirates': 'ARE',
    'UAE': 'ARE',
    'ARE': 'ARE',
    
    'Turkey': 'TUR',
    'TUR': 'TUR',
    
    'Syria': 'SYR',
    'Syrian Arab Republic': 'SYR',
    'SYR': 'SYR',
    
    'Jordan': 'JOR',
    'JOR': 'JOR',
    
    'Lebanon': 'LBN',
    'LBN': 'LBN',
    
    'Israel': 'ISR',
    'ISR': 'ISR',
    
    'Palestine': 'PSE',
    'PSE': 'PSE',
    
    'Kyrgyzstan': 'KGZ',
    'KGZ': 'KGZ',
    
    'Tajikistan': 'TJK',
    'TJK': 'TJK',
    
    'Uzbekistan': 'UZB',
    'UZB': 'UZB',
    
    'Turkmenistan': 'TKM',
    'TKM': 'TKM',
    
    # 欧洲国家
    'Spain': 'ESP',
    'ESP': 'ESP',
    
    'Portugal': 'PRT',
    'PRT': 'PRT',
    
    'France': 'FRA',
    'FRA': 'FRA',
    
    'Italy': 'ITA',
    'ITA': 'ITA',
    
    'Greece': 'GRC',
    'GRC': 'GRC',
    
    'Albania': 'ALB',
    'ALB': 'ALB',
    
    'Bulgaria': 'BGR',
    'BGR': 'BGR',
    
    'Romania': 'ROU',
    'ROU': 'ROU',
    
    'Ukraine': 'UKR',
    'UKR': 'UKR',
    
    'Belarus': 'BLR',
    'BLR': 'BLR',
    
    'Poland': 'POL',
    'POL': 'POL',
    
    'Germany': 'DEU',
    'DEU': 'DEU',
    
    'Sweden': 'SWE',
    'SWE': 'SWE',
    
    'Finland': 'FIN',
    'FIN': 'FIN',
    
    'Norway': 'NOR',
    'NOR': 'NOR',
    
    # 其他
    'New Zealand': 'NZL',
    'NZL': 'NZL',
    
    'Fiji': 'FJI',
    'FJI': 'FJI',
    
    'Vanuatu': 'VUT',
    'VUT': 'VUT',
    
    'Solomon Islands': 'SLB',
    'SLB': 'SLB',
    
    'New Caledonia': 'NCL',
    'NCL': 'NCL',
    
    'French Polynesia': 'PYF',
    'PYF': 'PYF',
}


def country_to_iso3(country_name: str) -> str:
    """
    将国家名称转换为ISO3代码
    
    Args:
        country_name: 国家名称（字符串）
    
    Returns:
        ISO3代码（字符串），如果找不到则返回None
    """
    if pd.isna(country_name) or country_name is None:
        return None
    
    country_str = str(country_name).strip()
    
    # 直接查找
    if country_str in COUNTRY_TO_ISO3:
        return COUNTRY_TO_ISO3[country_str]
    
    # 尝试大小写不敏感查找
    country_lower = country_str.lower()
    for key, iso3 in COUNTRY_TO_ISO3.items():
        if key.lower() == country_lower:
            return iso3
    
    # 如果找不到，返回None
    return None


def add_iso3_to_dataframe(df: pd.DataFrame, country_column: str = 'country') -> pd.DataFrame:
    """
    向DataFrame添加iso3列
    
    Args:
        df: 包含country列的DataFrame
        country_column: country列的名称，默认为'country'
    
    Returns:
        添加了iso3列的DataFrame（原地修改）
    """
    if country_column not in df.columns:
        print(f"⚠️  DataFrame中没有'{country_column}'列，无法添加iso3列")
        return df
    
    # 如果iso3列已存在，先检查是否需要更新
    if 'iso3' in df.columns:
        # 检查是否所有值都是NaN或空
        existing_iso3 = df['iso3'].dropna()
        existing_iso3 = existing_iso3[existing_iso3 != '']
        if len(existing_iso3) > 0:
            print(f"✅ DataFrame已有iso3列，且包含 {len(existing_iso3)} 个有效值")
            return df
    
    # 添加iso3列
    df['iso3'] = df[country_column].apply(country_to_iso3)
    
    # 统计转换结果
    iso3_counts = df['iso3'].dropna()
    iso3_counts = iso3_counts[iso3_counts != '']
    print(f"✅ 成功添加iso3列: {len(iso3_counts)} / {len(df)} 行有有效的ISO3代码")
    
    if len(iso3_counts) < len(df) * 0.5:
        print(f"⚠️  警告：只有 {len(iso3_counts)/len(df)*100:.1f}% 的行有有效的ISO3代码")
        # 显示一些未匹配的国家名称
        unmatched = df[df['iso3'].isna() | (df['iso3'] == '')][country_column].unique()[:10]
        if len(unmatched) > 0:
            print(f"   未匹配的国家名称示例: {list(unmatched)}")
    
    return df

