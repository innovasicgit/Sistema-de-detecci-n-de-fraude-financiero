function focusNavigationItem(item) {
  document.querySelectorAll('[data-navigation-item]').forEach((element) => {
    element.removeAttribute('data-active');
  });
  item.setAttribute('data-active', 'true');
}

function announce(message) {
  const region = document.querySelector('[aria-live="polite"]');
  if (region) region.textContent = message;
}
