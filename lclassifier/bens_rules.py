


    elements = ['\bbegin\b', '\bend\b', '\bdo\b', '\bvar\b', '\bdefine\b', '\bdefn\b', '\bfunction\b',
                '\bclass\b', '\bmy\b', '\brequire\b', '\bvoid\b', '\bval\b', '\bpublic\b', '\blet\b',
                '\bwhere\b', '\busing\b', '\bextend\b', '\bfunction\b']
    results = []
    for element in elements:
        results.append(len(re.findall(element, text)))

    elements = ['[)]+','[}]+', '[\]]+', '[=]+']

    for element in elements:
        runs = sorted(re.findall(element, text), key=len)
        if runs:
            results.append(len(runs[-1]))
        else:
            results.append(0)
    return results
