import json

# Load the MITRE ATT&CK data
with open('mitre_enterprise/enterprise-attack.json', 'r') as f:
    mitre_data = json.load(f)

def get_tactic_info(tactic_name):
    for tactic in mitre_data['objects']:
        if tactic['type'] == 'x-mitre-tactic' and tactic['name'].lower() == tactic_name.lower():
            return {
                "name": tactic['name'],
                "description": tactic['description']
            }
    return None

def get_technique_info(technique_id):
    for technique in mitre_data['objects']:
        if technique['type'] == 'attack-pattern' and technique['external_references'][0]['external_id'] == technique_id:
            return {
                "name": technique['name'],
                "description": technique['description'],
                "mitigation": get_mitigation_for_technique(technique_id)
            }
    return None

def get_mitigation_for_technique(technique_id):
    mitigations = []
    for mitigation in mitre_data['objects']:
        if mitigation['type'] == 'course-of-action':
            for ref in mitigation.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack' and ref.get('external_id') == technique_id:
                    mitigations.append(mitigation['description'])
    return mitigations

def enrich_analysis(analysis):
    enriched = {}
    for tactic, count in analysis['tactic_distribution'].items():
        tactic_info = get_tactic_info(tactic)
        if tactic_info:
            enriched[tactic] = {
                "count": count,
                "info": tactic_info
            }

    for technique, count in analysis['technique_distribution'].items():
        technique_info = get_technique_info(technique)
        if technique_info:
            if technique not in enriched:
                enriched[technique] = {}
            enriched[technique].update({
                "count": count,
                "info": technique_info
            })

    return enriched