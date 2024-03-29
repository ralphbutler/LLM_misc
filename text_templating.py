
## withOUT var names in the template ###

template = "{} was born in {}."
names_years = [("Alice", 1980), ("Bob", 1992), ("Charlie", 1975)]

for (name,year) in names_years:
    print(template.format(name, year))

print("-"*50)

## with var names in the template ###

template = "{name} was born in {year}."
names_years = [("Alice", 1980), ("Bob", 1992), ("Charlie", 1975)]

for (name,year) in names_years:
    print(template.format(name=name, year=year))

print("-"*50)

## using Template from string ###

from string import Template

template = Template("$name was born in $year")
names_years = [("Alice", 1980), ("Bob", 1992), ("Charlie", 1975)]

for (name,year) in names_years:
    print(template.substitute(name=name,year=year))
