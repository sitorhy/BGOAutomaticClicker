import {Expect, PathVariables, RequestBody, RequestConfig, RequestMapping, Service} from "axios-annotations";
import {config} from "./config.ts";

@RequestConfig(config)
export default class ResourcesApi extends Service {

    @RequestMapping("/res/list/{dir}", "GET")
    @PathVariables()
    getAssetsImages(dir: string) {
        return Expect<{
            dir: string
            files: {
                name: string
                path: string
                url: string
            }[]
            subdirs: string[]
        }>({
            dir
        });
    }


    @RequestMapping("/detect/avatar", "POST")
    @RequestBody()
    detectAvatarRect(mask_pic_ath: string, template_pic_path: string, target_pic_path: string) {
        return Expect<{
            scale: number,
            template_size: [number, number],
            score: number,
            x: number,
            y: number,
            w: number,
            h: number,
            rect: [number, number, number, number],
        }[]>({
            body: {
                mask_pic_ath,
                template_pic_path,
                target_pic_path,
            }
        });
    }
}