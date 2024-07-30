document.addEventListener('DOMContentLoaded', function() {
    const modeButtons = document.querySelectorAll('.mode-selection .btn');
    modeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const mode = this.getAttribute('data-mode');
            if (mode === 'checkout') {
                window.location.href = '/checkout';
            } else if (mode === 'employee') {
                window.location.href = '/login';
            }
        });
    });
});
