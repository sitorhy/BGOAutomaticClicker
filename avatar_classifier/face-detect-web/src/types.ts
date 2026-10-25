export interface Servant {
  id: number
  name_cn: string
  name_jp: string
  name_en: string
  page_url: string
  avatar: string
  card_rarity: string
  servant_class: string
  np_color: string
  np_type: string
  cards: string
  faction: string
  get: string
  atk_max: number
  hp_max: number
  revoke: number
}

export interface ServantAvatar {
  id: number
  servant_id: number
  file: string
  left: number
  top: number
  right: number
  bottom: number
}

export interface BBoxRange {
  left: number
  top: number
  right: number
  bottom: number
}

export type QueueStep = 'idle' | 'fetching_images' | 'fetching_avatars'

export interface ServantStats {
  imageCount: number
  markedCount: number
  step: QueueStep
}
