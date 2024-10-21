function calculateDowry() {
    let price = 100;

    const education = document.getElementById('education').value;
    const networth = document.getElementById('networth').value;
    const caste = document.getElementById('caste').value;
    const skills = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
        .map(skill => skill.value);
    const age = document.querySelector('input[name="age"]:checked').value;
    const reputation = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
        .map(rep => rep.value);


    const educationCoefficients = {
        bachelor: 1.5,
        college: 1.2,
        high_school: 1.05,
        middle_school: 0.9
    };
    if (educationCoefficients[education]) {
        price *= educationCoefficients[education];
    }

    const networthCoefficients = {
        upper_class: 2,
        middle_class: 1.5,
        lower_class: 1.2
    };
    if (networthCoefficients[networth]) {
        price *= networthCoefficients[networth];
    }

    const casteValues = {
        brahmin: 100,
        kshatriya: 50,
        vaishya: 20,
        shudra: 10,
        untouchable: -50
    };
    if (casteValues[caste]) {
        price += casteValues[caste];
    }

    const skillValues = {
        musician: 10,
        cook: 20,
        easygoing: 15,
        singer: 10
    };
    skills.forEach(skill => {
        if (skillValues[skill]) {
            price += skillValues[skill];
        }
    });

    const ageCoefficients = {
        young: 1.5,
        middle: 1.2,
        older: 0.95
    };
    if (ageCoefficients[age]) {
        price *= ageCoefficients[age];
    }

    const reputationValues = {
        parent_gossip: 0.85,
        character_gossip: 0.9,
        general_gossip: -20
    };
    reputation.forEach(rep => {
        if (reputationValues[rep]) {
            if (rep === 'general_gossip') {
                price += reputationValues[rep];
            } else {
                price *= reputationValues[rep];
            }
        }
    });

    document.getElementById('result').innerText = `Final Dowry Price: $${price.toFixed(2)}`;
}

document.getElementById('submit').addEventListener('click', calculateDowry);
