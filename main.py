import time

from main_controller import (
    ControllerConfig,
    MainController,
    _display_sort_key as _controller_display_sort_key,
    download_live_images_local as _download_live_images_local_impl,
    download_live_images_remote as _download_live_images_remote_impl,
    process_remote_event as _process_remote_event_impl,
)
from settings import get_optional_sftp_settings
from utilities.log import get_logger, install_print_logger


def _display_sort_key(filename):
    return _controller_display_sort_key(filename)


def _download_live_images_local(file_manager, local_path, rotation_state, logger, max_images=7):
    return _download_live_images_local_impl(
        file_manager=file_manager,
        local_path=local_path,
        rotation_state=rotation_state,
        logger=logger,
        max_images=max_images,
    )


def _download_live_images_remote(
    app,
    remote_path,
    local_path,
    remote_hist_dir,
    rotation_state,
    logger,
    max_images=7,
):
    return _download_live_images_remote_impl(
        app=app,
        remote_path=remote_path,
        local_path=local_path,
        remote_hist_dir=remote_hist_dir,
        rotation_state=rotation_state,
        logger=logger,
        max_images=max_images,
    )


def _process_remote_event(msg, display, logger):
    return _process_remote_event_impl(msg=msg, display=display, logger=logger)


def main():
    install_print_logger(reset=True)
    logger = get_logger()

    from display_window import DisplayWindow
    from sftp_app import SFTPApp

    try:
        sftp_credentials = get_optional_sftp_settings()
    except Exception as exc:
        logger.error(
            f"[SSH] Invalid SFTP settings, running local-only: {exc}",
            allow_repeat=True,
        )
        sftp_credentials = None

    sftp_app = None
    if sftp_credentials is not None:
        sftp_app = SFTPApp(
            sftp_credentials["hostname"],
            sftp_credentials["port"],
            sftp_credentials["username"],
            sftp_credentials["password"],
        )

    try:
        display = DisplayWindow(
            width=1920,
            height=1080,
            window_name="Display Imagenes",
            refresh_interval=0.25,
            sftp_client=sftp_app.sftp_client if sftp_app else None,
            sftp_credentials=sftp_credentials,
        )
        logger.info("[DB] DisplayWindow initialized successfully", allow_repeat=True)
    except Exception as exc:
        logger.error(f"[DB] Failed to initialize DisplayWindow: {exc}", allow_repeat=True)
        if sftp_app:
            sftp_app.disconnect_sftp()
        return

    controller = MainController(
        display=display,
        logger=logger,
        sftp_credentials=sftp_credentials,
        sftp_app=sftp_app,
        config=ControllerConfig(),
    )
    display.start_historic_download_on_startup(
        controller.config.temp_dir,
        check_interval=controller.config.historic_download_check_interval,
    )
    controller.run()


if __name__ == "__main__":
    main()
