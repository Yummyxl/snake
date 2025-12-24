export function sortEvalRollouts(items) {
  const list = Array.isArray(items) ? items.slice() : [];
  return list.sort((a, b) => {
    const ab = a?.is_best ? 1 : 0;
    const bb = b?.is_best ? 1 : 0;
    if (ab !== bb) return bb - ab;
    return Number(b?.created_at_ms || 0) - Number(a?.created_at_ms || 0);
  });
}

