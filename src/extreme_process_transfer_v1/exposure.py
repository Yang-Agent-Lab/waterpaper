"""Paired country/CN slots and foreign-only change of measure."""
import numpy as np
COUNTRIES = ('CN', 'DE', 'IS', 'US', 'JP')
MASSES = np.array([.50, .20, .10, .15, .05])


def probabilities(rows, base, severity):
    base = np.asarray(base, dtype=np.float64)
    severity = np.asarray(severity)
    countries = np.asarray([r['country'] for r in rows])
    if len(rows) != len(base) or len(base) != len(severity):
        raise ValueError('unaligned fit identities')
    changed = base.copy()
    for country, mass in zip(COUNTRIES, MASSES):
        ids = np.flatnonzero(countries == country)
        np.testing.assert_allclose(base[ids].sum(), mass, atol=1e-12, rtol=0)
        if country == 'CN':
            continue
        classes = np.unique(severity[ids])
        for category in classes:
            ix = ids[severity[ids] == category]
            class_mass = base[ix].sum()
            if class_mass <= 0:
                raise ValueError('zero class probability')
            changed[ix] = .5 * base[ix] + .5 * mass / len(classes) * base[ix] / class_mass
        np.testing.assert_allclose(changed[ids].sum(), mass, atol=1e-12, rtol=0)
    if np.any(base <= 0) or np.any(changed <= 0):
        raise ValueError('fit support is not strictly positive')
    np.testing.assert_allclose(changed.sum(), 1., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(changed[countries == 'CN'], base[countries == 'CN'])
    return changed, countries


def paired_schedule(rows, base, severity, steps=3000, batch=64, seed=2):
    changed, countries = probabilities(rows, base, severity)
    generator = np.random.default_rng(seed)
    slots = generator.choice(len(COUNTRIES), size=(steps, batch), p=MASSES)
    natural = np.empty(slots.shape, dtype=np.int64)
    balanced = np.empty_like(natural)
    for ci, (country, mass) in enumerate(zip(COUNTRIES, MASSES)):
        ids = np.flatnonzero(countries == country)
        locations = np.flatnonzero(slots.ravel() == ci)
        natural.ravel()[locations] = generator.choice(ids, len(locations), p=base[ids] / mass)
        if country == 'CN':
            balanced.ravel()[locations] = natural.ravel()[locations]
        else:
            balanced.ravel()[locations] = generator.choice(ids, len(locations), p=changed[ids] / mass)
    weight = np.asarray(base) / changed
    np.testing.assert_allclose(changed * weight, base, atol=1e-17, rtol=1e-14)
    return natural, balanced, weight, slots, changed
