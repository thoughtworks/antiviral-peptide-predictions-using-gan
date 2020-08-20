other_alphabets = "UOXBZJ"


def contains(other_alphabets, seq):
    for o in str(other_alphabets):
        if o in str(seq):
            return True
    return False

def additional_filtering_of_sequences(data):
    sequences = data[data.apply(lambda r: not contains(other_alphabets, r['Sequence']), axis=1)]
    sequences = sequences[sequences.apply(lambda r: not str(r['Sequence']) == 'nan', axis=1)]
    sequences['Sequence'] = sequences['Sequence'].apply(lambda x: x.upper())
    indexNames = sequences[sequences['Sequence'] == 'GWLDVAKKIGKAAFNVAKNFLFNKAVNFAAKGIKKAVDLWG '].index
    sequences.drop(indexNames, inplace=True)
    sequences = sequences[sequences.Sequence.apply(lambda x: x.isalpha())]

    return sequences
