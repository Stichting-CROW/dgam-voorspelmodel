# Datagedreven assetmanagement voorspelmodel 
Ontwikkeld door het Big Data Innovatiehub van de Haagse Hogeschool, onder begeleiding van Dr. Raymond Hoogendoorn. 

### Datalab
Het datalab (ook wel Big Data Innovatiehub genoemd) brengt de kennis, vaardigheden, data en technologie samen om daadwerkelijk opdrachten te kunnen uitvoeren. Online is het nieuwe normaal geworden op veel verschillende levensterreinen – van werken en leren tot ontspannen. De verwachting is dat de digitale transformatie zich steeds verder zal voortzetten. Dit vraagt van ondernemers om zich aan te passen. Het optimaal gebruik van data is daarin een voorwaarde. Maar niet alle mkb’ers hebben de mogelijkheid of middelen om hun vraagstukken zelf op te lossen.

### Werking van de web app en Azure omgeving
Er kan worden ingelogd op een van de accounts voor de gemeenten op de site https://voorspelmodel-webapp.azurewebsites.net/ . Op de pagina van de gemeente kunnen eerdere bestanden worden gedownload. Ook kan er worden gekozen om een nieuw bestand te uploaden. Dit bestand komt in onze Azure omgeving terecht. Via een data factory pipeline wordt dit bestand in onze databricks omgeving verwerkt. Met het getrainde model voegen we een voorspelling toe en wordt het nieuwe bestand onderaan beschikbaar om te downloaden.
In het bestand hoeven niet alle kolommen te zitten waarop getraind is. Hoe meer hoe beter, in het vakje voor missing features is te zien welke kolommen niet in het bestand.

### Data
Er kan worden ingelogd op een van de accounts voor de gemeenten. Op de pagina van de gemeente kunnen eerdere bestanden worden gedownload. Ook kan er worden gekozen om een nieuw bestand te uploaden. Dit bestand komt in onze Azure omgeving terecht. Via een data factory pipeline wordt dit bestand in onze databricks omgeving verwerkt. Met het getrainde model voegen we een voorspelling toe en wordt het nieuwe bestand onderaan beschikbaar om te downloaden.
