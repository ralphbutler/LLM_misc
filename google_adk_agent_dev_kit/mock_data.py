"""
Mock data for the Earth shape debate tools.
"""

# Evidence data with both perspectives
EVIDENCE_DATA = {
    "horizon": {
        "round_earth": {
            "fact": "Ships disappear from the bottom up when sailing away, consistent with Earth's curvature.",
            "explanation": "As ships sail away, they appear to sink below the horizon because of Earth's curvature. This has been observed for centuries and is consistent with a spherical Earth."
        },
        "flat_earth": {
            "fact": "Ships appear to sink due to perspective and the limitations of human vision.",
            "explanation": "The disappearance of ships is due to perspective and the limits of visual acuity. With a powerful enough telescope, ships can be brought back into view after they've 'disappeared'."
        }
    },
    "gravity": {
        "round_earth": {
            "fact": "Gravity pulls objects toward the center of the Earth, creating a spherical planet.",
            "explanation": "The nearly uniform gravitational pull toward Earth's center causes large masses to form spheres. This is why all celestial bodies above a certain size are spherical."
        },
        "flat_earth": {
            "fact": "Objects fall downward due to density, not gravity.",
            "explanation": "Things fall because they are denser than the air around them. This universal acceleration upward creates the effect we mistake for gravity."
        }
    },
    "photos": {
        "round_earth": {
            "fact": "Numerous photos from space show Earth as a sphere.",
            "explanation": "Since the 1960s, thousands of photographs from space missions show Earth as a sphere. These come from multiple space agencies and private companies."
        },
        "flat_earth": {
            "fact": "Space photos are manipulated or created with CGI.",
            "explanation": "NASA and other space agencies use composite images, fish-eye lenses, and CGI to create the curved appearance in supposed 'photos' of Earth."
        }
    },
    "time_zones": {
        "round_earth": {
            "fact": "Time zones exist because the sun illuminates different parts of the spherical Earth at different times.",
            "explanation": "As the Earth rotates, different longitudes face the sun at different times, creating our time zones. This is only possible on a spherical planet."
        },
        "flat_earth": {
            "fact": "The sun moves in circles above the flat Earth, creating the appearance of time zones.",
            "explanation": "The sun acts like a spotlight, illuminating different areas as it moves in a circular path above the flat Earth, creating the effect of time zones."
        }
    },
    "circumnavigation": {
        "round_earth": {
            "fact": "People have circumnavigated the Earth in all directions, confirming its spherical shape.",
            "explanation": "Since Magellan's expedition in the 16th century, people have sailed and flown around the world in various directions, which is only possible on a sphere."
        },
        "flat_earth": {
            "fact": "Circumnavigation works on a flat Earth by traveling in a circle around the North Pole.",
            "explanation": "The flat Earth model has the North Pole at the center and Antarctica as a rim. Circumnavigation just means traveling in a circle around the central point."
        }
    }
}

# Historical references data
HISTORICAL_REFERENCES = {
    "ancient_greece": {
        "round_earth": {
            "reference": "Eratosthenes calculated Earth's circumference in 240 BCE using shadow measurements.",
            "details": "By measuring shadows in different cities at the same time, Eratosthenes calculated Earth's circumference with remarkable accuracy for the time, proving Earth's curvature."
        },
        "flat_earth": {
            "reference": "Many early cultures depicted Earth as a flat disk in their cosmologies.",
            "details": "Various ancient cultures, including some early Greek philosophers before Aristotle, depicted Earth as a flat disk in their cosmological models."
        }
    },
    "medieval_period": {
        "round_earth": {
            "reference": "Medieval scholars like Thomas Aquinas accepted Earth's sphericity.",
            "details": "Contrary to popular belief, educated people in medieval Europe accepted that Earth was a sphere, building on ancient Greek knowledge."
        },
        "flat_earth": {
            "reference": "Religious interpretations sometimes described Earth with 'four corners'.",
            "details": "Some literal interpretations of religious texts described Earth as having corners or ends, supporting a non-spherical model."
        }
    },
    "modern_era": {
        "round_earth": {
            "reference": "The 1957 Sputnik satellite and subsequent space missions directly observed Earth's curvature.",
            "details": "The space age provided direct observation of Earth's spherical shape through photographs and satellite data."
        },
        "flat_earth": {
            "reference": "The modern Flat Earth Society was founded in the 1950s by Samuel Shenton.",
            "details": "The modern flat Earth movement gained new followers in the internet age, questioning established scientific consensus."
        }
    }
}

# Image reference data
IMAGE_REFERENCES = {
    "space_photos": {
        "round_earth": {
            "image": "Blue Marble (1972)",
            "description": "Famous NASA photograph of Earth taken by Apollo 17 astronauts, showing a fully illuminated sphere."
        },
        "flat_earth": {
            "image": "Azimuthal Equidistant Projection",
            "description": "The UN logo-style map projection that flat-Earthers claim represents the actual flat Earth layout with the North Pole at center."
        }
    },
    "horizon_test": {
        "round_earth": {
            "image": "Lake Pontchartrain Power Lines",
            "description": "Photographs showing power lines crossing Lake Pontchartrain that visibly curve with Earth's surface."
        },
        "flat_earth": {
            "image": "Bedford Level Experiment Recreation",
            "description": "Recreation of the 19th century experiment on the Old Bedford River that flat-Earthers claim shows no curvature over long distances."
        }
    },
    "experiments": {
        "round_earth": {
            "image": "Foucault Pendulum",
            "description": "Pendulum experiment demonstrating Earth's rotation, which aligns with a spherical Earth model."
        },
        "flat_earth": {
            "image": "Laser Test Across Still Water",
            "description": "Laser beam tests across still water that flat-Earthers claim show no curvature over distances where it should be measurable."
        }
    }
}
