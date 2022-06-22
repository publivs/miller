import pandas as pd

node = {"children": [
			{
				"children": [
					{
						"children": [
							{
								"dicJsonDtoList": [
									{
										"dictionaryCodeName": "50502 股票——财富管理",
										"dictionaryCode": "505020101"
									},
									{
										"dictionaryCodeName": "50501 平衡型——财富管理",
										"dictionaryCode": "505010101"
									}
								],
								"tagCodeDet": "DETAIL_ASSET_CODE_NEW_LABEL",
								"children": [],
								"opSignDet": "in",
								"rel": " ",
								"dictionaryCode": [
									"505020101",
									"505010101"
								]
							},
							{
								"dicJsonDtoList": [
									{
										"dictionaryCodeName": "CNY",
										"dictionaryCode": "1"
									}
								],
								"tagCodeDet": "CURRENCY_CD_NEW_LABEL",
								"opValueDetText": "",
								"children": [],
								"opSignDet": "in",
								"rel": "and",
								"dictionaryCode": [
									"1"
								]
							},
							{
								"prefix": " case when",
								"suffix": " then false else true end ",
								"children": [
									{
										"children": [
											{
												"dicJsonDtoList": [
													{
														"dictionaryCodeName": "601398.SH-工商银行",
														"dictionaryCode": "601398.SH"
													},
													{
														"dictionaryCodeName": "601229.SH-上海银行",
														"dictionaryCode": "601229.SH"
													},
													{
														"dictionaryCodeName": "A02105.PF-大盘价值资产管理计划",
														"dictionaryCode": "A02105.PF"
													},
													{
														"dictionaryCodeName": "A02160.PF-广发特定策略1号特定多客户资产管理计划",
														"dictionaryCode": "A02160.PF"
													},
													{
														"dictionaryCodeName": "A02129.PF-泓德金融蓝筹1号资产管理计划",
														"dictionaryCode": "A02129.PF"
													},
													{
														"dictionaryCodeName": "A02132.PF-泓德金融蓝筹2号资产管理计划",
														"dictionaryCode": "A02132.PF"
													},
													{
														"dictionaryCodeName": "A02133.PF-泓德金融蓝筹3号资产管理计划",
														"dictionaryCode": "A02133.PF"
													},
													{
														"dictionaryCodeName": "A02176.PF-汇添富-添富牛53号资产管理计划",
														"dictionaryCode": "A02176.PF"
													}
												],
												"tagCodeDet": "TAA_SECURITY_ID",
												"opValueDetText": "",
												"children": [],
												"opSignDet": "in",
												"rel": " ",
												"isCopy": "",
												"dictionaryCode": [
													"601398.SH",
													"601229.SH",
													"A02105.PF",
													"A02160.PF",
													"A02129.PF",
													"A02132.PF",
													"A02133.PF",
													"A02176.PF"
												]
											}
										]
									}
								],
								"rel": "and",
								"logicalRelation": ""
							}
						],
						"rel": " "
					}
				]
			},
			{
				"tagCodeDet": "NODE_ID",
				"opSignDet": "in",
				"rel": "and",
				"dictionaryCode": [
					"N00000004"
				]
			}
		]}

# 陈证码
# 证
# ＃.遍历找到字典的所有路径
def dic2sql(node):
    if not node.get('children'):
        node['opSignDet'] = node["opSignDet"].rstrip()

        if node["opSignDet"] == "like" or node["opSignDet"] == 'not like':
            dictionaryCode = f"'{node['dictionaryCode'][0]}'"
        elif node["opSignDet"] == "pre like":
            node["opSignDet"] = "like"
            dictionaryCode = f"'{node['dictionaryCode'][0][2:]}'"
        elif node['opSignDet'] == 'suf like':
            node["opSignDet"] = "like"
            dictionaryCode = f"'{node['dictionaryCode'][0][:-2]}'"
        elif node['opSignDet'] == 'in' or node['opSignDet'] == 'not in':
            dictionaryCode = ','.join(["'%s'" % item for item in node['dictionaryCode']])
            dictionaryCode = f'({dictionaryCode})'
        elif node['opSignDet'] == 'is':
            dictionaryCode = "null"
            return f"{node['rel']} (({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode}) or ({node['tagCodeDet']} = ''))"
        elif node['opSignDet'] == "is not":
            dictionaryCode ="null"
            return f"{node['rel']} (({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode})or ({node['tagCodeDet']} != ''))"
        elif node['opsignDet'] in ['>','<','>=','<=']:
            dictionaryCode=node['dictionarycode'][0]
            return f"{node['rel']} ({node['tagCodeDet']} {node['opSignDet']} '{dictionaryCode}')"
        else:
            dictionaryCode = node['dictionaryCode']
        return f"{node['rel']} ({node['tagCodeDet']} {node['opSignDet']} {dictionaryCode})"

    res= ' '
    tag = node.get('rel','')
    prefix = node.get("prefix")
    suffix = node.get("suffix")

    for i in node.get('children'):

        res = f"{res} {dic2sql(i)}"

    if prefix:
        res = f"{tag} {prefix} ({res}) {suffix}"
    else:
        res = f"{tag} ({res})"

    return res


res = dic2sql(node)
print(res)
