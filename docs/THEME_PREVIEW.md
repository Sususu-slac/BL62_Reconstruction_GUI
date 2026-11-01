# Light / Dark theme preview

The approved colors are now applied to BL62 Reconstruction, BL62 Label GUI,
BL62 Fine-tune GUI and MomoBot. Each application remembers its own theme.

The current preview uses blue accents in Light mode and the original green
accents in Dark mode, following the user's color review. Primary buttons,
selected controls, focus borders, progress bars and status accents follow each
theme. Reference-mask colors and image data remain unchanged.
The previous green palette is retained in
`test_outputs/theme_green_palette_20261101.py`; blue captures are in
`test_outputs/theme_preview_blue_20261101`.

Run the existing source application with `--theme-preview` to open the interactive
preview. The window title ends in **Theme Preview**; its initial theme is Light.
The upper-right button names its destination: **Switch to Light** in Dark mode,
and **Switch to Dark** in Light mode. It switches immediately. Both the
Reconstruction and Intelligent Data Analysis pages use the same theme, as do
settings, reference-mask editing, menus, logs, file pickers and progress bars.

The regular application's first-launch default remains Dark. Theme selection is
remembered. The preview uses `theme_preview.json` beside the normal settings file,
so reviewing colors does not change the desktop version's stored settings. Saving
a theme merges only that preference into the file, preserving other saved options
and leaving unsaved task settings alone.

Images, masks, reference outlines and saved data retain their original content.
Image-view margins use a neutral gray in Light mode. Score-chart backgrounds,
axes, tick text, curve colors and markers change without clearing measured points
or changing the displayed range. Theme switching remains available during a job.

The preview uses Qt file dialogs so their fields and file lists follow the theme.
It does not change the Windows system theme or any other application's colors.

The desktop builds now include these colors. Reconstruction 1.2.0 and Fine-tune
1.1.0 installers are available locally. Label and MomoBot desktop launchers load
the updated source. Existing scientific backend code is unchanged by the theme feature.

Visual checks: both main pages, both themes, analysis settings, reference-mask
editor, and Qt file picker. Preview captures are in
`test_outputs/theme_preview_20261101`; the chart/image examples shown in those
captures are previous saved results, not new reconstruction runs.
