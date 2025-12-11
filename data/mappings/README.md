# Drug ID Mapping

The file `drug-mappings.tsv` (see [source](https://github.com/iit-Demokritos/drug_id_mapping/blob/main/drug-mappings.tsv)) lists thousands of known drugs and IDs available in drug databases. Please see [iit-Demokritos/drug_id_mapping](https://github.com/iit-Demokritos/drug_id_mapping) for more information.

The authors retrieved drug information included in Drugbank (version 5.1.8, released on 2021-01-03) as well as in the latest Therapeutic Target Database (version 7.1.01, released on 2019.07.14) in a file. Then, they add identifiers from the following sources:
*  The web services API of ChEMBL Database.
*  The PUG REST API of PubChem Database.
*  The drugs file in the FTP server of the KEGG Database.
*  The UMLS Metathesaurus vocabulary Database, using the MetamorphoSys tool.
*  The mapping files of the STITCH Database.

For more information, please see:
> Aisopos, F., Paliouras, G. Comparing methods for drug–gene interaction prediction on the biomedical literature knowledge graph: performance versus explainability. BMC Bioinformatics 24, 272 (2023), doi: [10.1186/s12859-023-05373-2](https://doi.org/10.1186/s12859-023-05373-2).


# MONDO to EFO Mapping

The file `mondo_efo_mappings.tsv` (see [source](https://github.com/EBISPOT/efo/blob/master/src/ontology/components/mondo_efo_mappings.tsv)) lists mappings from Monarch Disease Ontology (MONDO) IDs to Experimental Factor Ontology (EFO) IDs. See [issue #1381](https://github.com/EBISPOT/efo/issues/1381) in the EFO repository on GitHub for more information.


# HGNC Approved Gene Symbols

The file `hgnc_complete_set.txt` (see [source](https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt)) was downloaded from the [Human Gene Nomenclature Committee](https://www.genenames.org/download/statistics-and-files/) to resolve any conflicting gene IDs.
