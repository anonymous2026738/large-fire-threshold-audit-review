"""
ISO3
forFireTrackscountryiso3
"""
import pandas as pd

# ISO3
# 
COUNTRY_TO_ISO3 = {
    # ()
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
    
    # 
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
    
    # 
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
    
    # 
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
    
    # 
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
    ISO3
    
    Args:
        country_name: ()
    
    Returns:
        ISO3(),None
    """
    if pd.isna(country_name) or country_name is None:
        return None
    
    country_str = str(country_name).strip()
    
    # 
    if country_str in COUNTRY_TO_ISO3:
        return COUNTRY_TO_ISO3[country_str]
    
    # 
    country_lower = country_str.lower()
    for key, iso3 in COUNTRY_TO_ISO3.items():
        if key.lower() == country_lower:
            return iso3
    
    #,None
    return None


def add_iso3_to_dataframe(df: pd.DataFrame, country_column: str = 'country') -> pd.DataFrame:
    """
    DataFrameiso3
    
    Args:
        df: countryDataFrame
        country_column: country,'country'
    
    Returns:
        iso3DataFrame()
    """
    if country_column not in df.columns:
        print(f"⚠️  DataFrame'{country_column}',iso3")
        return df
    
    # iso3,
    if 'iso3' in df.columns:
        # NaN
        existing_iso3 = df['iso3'].dropna()
        existing_iso3 = existing_iso3[existing_iso3 != '']
        if len(existing_iso3) > 0:
            print(f"✅ DataFrameiso3, {len(existing_iso3)} ")
            return df
    
    # iso3
    df['iso3'] = df[country_column].apply(country_to_iso3)
    
    # 
    iso3_counts = df['iso3'].dropna()
    iso3_counts = iso3_counts[iso3_counts != '']
    print(f"✅ iso3: {len(iso3_counts)} / {len(df)} ISO3")
    
    if len(iso3_counts) < len(df) * 0.5:
        print(f"⚠️  : {len(iso3_counts)/len(df)*100:.1f}% ISO3")
        # 
        unmatched = df[df['iso3'].isna() | (df['iso3'] == '')][country_column].unique()[:10]
        if len(unmatched) > 0:
            print(f"   : {list(unmatched)}")
    
    return df

