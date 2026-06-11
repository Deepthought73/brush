use crate::ui::panels::AppPane;
use crate::ui::ui_process::UiProcess;
use brush_process::message::{ProcessMessage, TrainMessage};
use eframe::emath::pos2;
use eframe::epaint::Color32;
use egui::{ColorImage, TextureOptions, Ui, WidgetText};
use image::RgbImage;

#[derive(Default)]
pub struct LastImagePanel {
    image: Option<RgbImage>,
}

impl AppPane for LastImagePanel {
    fn title(&self) -> WidgetText {
        let mut job = egui::text::LayoutJob::default();
        job.append(
            "Last Image",
            0.0,
            egui::TextFormat {
                color: Color32::WHITE,
                ..Default::default()
            },
        );
        job.into()
    }

    fn ui(&mut self, ui: &mut Ui, _: &UiProcess) {
        if let Some(img) = &self.image {
            let texture = {
                let size = [img.width() as usize, img.height() as usize];
                let color_image = ColorImage::from_rgb(size, img.as_raw());
                ui.load_texture("last-image", color_image, TextureOptions::default())
            };

            let available = ui.available_size();
            let cursor_min = ui.cursor().min;
            let aspect_ratio = texture.aspect_ratio();

            let mut size = available;
            if size.x / size.y > aspect_ratio {
                size.x = size.y * aspect_ratio;
            } else {
                size.y = size.x / aspect_ratio;
            }

            let offset_x = (available.x - size.x) / 2.0;
            let offset_y = (available.y - size.y) / 2.0;
            let min = cursor_min + egui::vec2(offset_x, offset_y);
            let rect = egui::Rect::from_min_size(min, size);

            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }
    }

    fn on_message(&mut self, message: &ProcessMessage, ui_process: &UiProcess) {
        if let ProcessMessage::TrainMessage(TrainMessage::NewImage { image }) = message {
            self.image = Some(image.to_rgb8());
            ui_process.repaint();
        }
    }
}
