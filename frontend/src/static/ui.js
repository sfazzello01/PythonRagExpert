(() => {
    const notices = document.getElementById('ui-notices');
    const dialog = document.getElementById('confirm-dialog');

    function notify(message, type = 'info') {
        if (!notices || !message) return;
        const notice = document.createElement('div');
        notice.className = `notice notice--${type}`;
        notice.setAttribute('role', type === 'error' ? 'alert' : 'status');
        notice.textContent = message;
        notices.appendChild(notice);
        window.setTimeout(() => notice.remove(), 5000);
    }

    function setBusy(button, busy, label) {
        if (!button) return;
        if (busy) {
            button.dataset.label = button.textContent;
            button.disabled = true;
            button.classList.add('is-busy');
            if (label) button.textContent = label;
            return;
        }
        button.disabled = false;
        button.classList.remove('is-busy');
        button.textContent = label || button.dataset.label || button.textContent;
    }

    function confirmAction({ title = 'Conferma azione', message = '', confirmLabel = 'Conferma' } = {}) {
        if (!dialog || typeof dialog.showModal !== 'function') {
            return Promise.resolve(window.confirm(message));
        }
        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-message').textContent = message;
        dialog.querySelector('button[value="confirm"]').textContent = confirmLabel;
        dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true });
        });
    }

    document.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            const form = event.target.closest('form');
            const submit = form?.querySelector('button[type="submit"]:not(:disabled)');
            if (submit) {
                event.preventDefault();
                submit.click();
            }
        }
    });

    window.UI = { notify, setBusy, confirmAction };
})();
