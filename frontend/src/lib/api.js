async function fetchJson(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url} answered ${response.status}`)
  return response.json()
}

/** Prefer the live API; fall back to the saved file so the dashboard also works as a static site. */
export async function loadResults() {
  try {
    return { data: await fetchJson('/api/results'), source: 'live API' }
  } catch {
    // fall through to the saved copy
  }
  try {
    return { data: await fetchJson('/results.json'), source: 'saved results' }
  } catch {
    throw new Error(
      'No results found. Run backend/scripts/export_results.py and copy experiments/export/results.json ' +
        'into frontend/public/.',
    )
  }
}